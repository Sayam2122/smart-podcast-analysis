import os
import sys
from data_loader import DataLoader
from vector_store import VectorStore
from query_engine import QueryEngine
from conversation_manager import ConversationManager
import re

EPISODES_DIR = os.path.join('data', 'episodes')
VECTOR_STORE_DIR = 'vector_store'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
OLLAMA_MODEL = 'mistral:latest'


def list_episodes():
    if not os.path.isdir(EPISODES_DIR):
        print(f"[quick_chat] Episodes directory not found: {EPISODES_DIR}")
        return []
    return [d for d in os.listdir(EPISODES_DIR) if os.path.isdir(os.path.join(EPISODES_DIR, d))]


def build_vector_store_for_episode(episode_id, loader):
    segments = loader.load_episode(episode_id)
    if not segments:
        print(f"[quick_chat] No data for episode: {episode_id}")
        return None, None, None
    vector_store = VectorStore(VECTOR_STORE_DIR, episode_id, EMBEDDING_MODEL)
    if not vector_store.is_built():
        print(f"Building vector store for {episode_id} (first time only)...")
        vector_store.build(segments)
    episode_model = loader.get_episode_model()
    return vector_store, segments, episode_model


def detect_intent(user_input):
    q = user_input.lower()
    if any(word in q for word in ['quote', 'shloka', 'shlok', 'sanskrit']):
        return 'quotes'
    if any(word in q for word in ['audio', 'mp3', 'wav', 'make audio', 'save audio']):
        return 'audio'
    return 'standard'


def detect_domain_and_guidance(episode_model, llm_client):
    # Use LLM to analyze the episode model and return a domain summary and system guidance
    prompt = f"""
You are a professional podcast domain analyst. Given the following episode model summary, identify:
- The most likely domain or topic of the episode(s) (e.g., spiritual, business, science, etc.)
- A 2-3 sentence summary of the domain and its key features
- How an AI assistant should process and answer questions for this domain (e.g., for spiritual, focus on philosophy, key figures, emotional tone, etc.)

EPISODE MODEL:
{episode_model}

---
Respond in this format:
Domain: <domain>
Summary: <summary>
Guidance: <guidance for answering professionally in this domain>
"""
    try:
        response = llm_client.generate(model=OLLAMA_MODEL, prompt=prompt)
        text = response['response'].strip()
        domain = summary = guidance = ''
        for line in text.splitlines():
            if line.lower().startswith('domain:'):
                domain = line.split(':', 1)[1].strip()
            elif line.lower().startswith('summary:'):
                summary = line.split(':', 1)[1].strip()
            elif line.lower().startswith('guidance:'):
                guidance = line.split(':', 1)[1].strip()
        return domain, summary, guidance
    except Exception as e:
        return 'Unknown', 'Could not detect domain.', f'Error: {e}'


def main():
    print("\n=== Podcast RAG Quick Chat ===")
    print("Type 'help' for commands. Type 'quit' to exit.\n")
    episodes = list_episodes()
    if not episodes:
        print("No episodes found. Please add episode folders with rag_ready.json.")
        return
    print("Available Episodes:")
    for i, ep in enumerate(episodes, 1):
        print(f"  {i}. {ep}")
    print("Type 'all' to search across all episodes, or comma-separated numbers for a subset.")
    while True:
        choice = input("\nSelect episode(s) [number,all,1,2,3]: ").strip()
        if choice.lower() == 'quit':
            return
        if choice.lower() == 'all':
            selected_episodes = episodes
            break
        try:
            idxs = [int(x.strip()) - 1 for x in choice.split(',')]
            if all(0 <= idx < len(episodes) for idx in idxs):
                selected_episodes = [episodes[idx] for idx in idxs]
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter numbers, 'all', or 'quit'.")

    loader = DataLoader(EPISODES_DIR)
    vector_stores = {}
    segments_map = {}
    episode_models = {}
    for eid in selected_episodes:
        vs, segs, emodel = build_vector_store_for_episode(eid, loader)
        if vs:
            vector_stores[eid] = vs
            segments_map[eid] = segs
            episode_models[eid] = emodel
    if not vector_stores:
        print("No valid episodes loaded.")
        return
    class MultiVectorStore:
        def __init__(self, stores):
            self.stores = stores
        def search(self, query, k=8):
            results = []
            for eid, store in self.stores.items():
                for r in store.search(query, k):
                    r['episode_id'] = eid
                    results.append(r)
            results.sort(key=lambda x: x.get('relevance_score', 9999))
            return results[:k]
    multi_vector_store = MultiVectorStore(vector_stores)
    def merge_episode_models(models):
        from collections import Counter
        all_themes = Counter()
        all_peaks = []
        all_speakers = Counter()
        all_key_moments = []
        for m in models.values():
            if m:
                all_themes.update(dict(m.get('main_themes', [])))
                all_peaks.extend(m.get('emotion_peaks', []))
                all_speakers.update(m.get('speaker_counts', {}))
                all_key_moments.extend(m.get('key_moments', []))
        return {
            'main_themes': all_themes.most_common(5),
            'emotion_peaks': all_peaks[:5],
            'speaker_counts': dict(all_speakers),
            'key_moments': all_key_moments[:10]
        }
    merged_episode_model = merge_episode_models(episode_models)
    all_segments = []
    for segs in segments_map.values():
        all_segments.extend(segs)
    query_engine = QueryEngine(multi_vector_store, OLLAMA_MODEL, merged_episode_model, all_segments)
    conv_manager = ConversationManager()
    session_id = conv_manager.start_new_session("_".join(selected_episodes))
    # --- Domain detection and system guidance ---
    domain, summary, guidance = detect_domain_and_guidance(merged_episode_model, query_engine.llm_client)
    print(f"\n[System] Detected Domain: {domain}")
    print(f"[System] Domain Summary: {summary}")
    print(f"[System] Guidance: {guidance}")
    print(f"\nQuick Chat session started: {session_id}")

    # --- Main Loop ---
    while True:
        user_input = input("\n💬 You: ").strip()
        if user_input.lower() in {'quit', 'exit'}:
            conv_manager.save_session()
            # Delete session file after save (simulate ephemeral chat)
            session_file = os.path.join(conv_manager.storage_dir, f"session_{session_id}.json")
            if os.path.exists(session_file):
                os.remove(session_file)
            print("Session ended and history deleted. Goodbye!")
            break
        if user_input.lower() == 'help':
            print("""
Commands:
  [question]         Ask anything about the episode(s)
  quotes             Extract quotes/shlokas
  audio              Extract audio segments and (stub) save audio file
  new                Start a new quick chat session
  quit/exit          Exit and delete this session's history
""")
            continue
        if user_input.lower() == 'new':
            print("\nStarting a new quick chat session...")
            conv_manager = ConversationManager()
            session_id = conv_manager.start_new_session("_".join(selected_episodes))
            # Redetect domain/guidance for new session
            domain, summary, guidance = detect_domain_and_guidance(merged_episode_model, query_engine.llm_client)
            print(f"\n[System] Detected Domain: {domain}")
            print(f"[System] Domain Summary: {summary}")
            print(f"[System] Guidance: {guidance}")
            print(f"New quick chat session started: {session_id}")
            continue
        intent = detect_intent(user_input)
        # Pass domain guidance to QueryEngine for use in prompts (if needed)
        query_engine.domain_guidance = guidance
        result = query_engine.ask(user_input, intent=intent)
        print(f"\n🤖 {result['answer']}")
        if result['suggestions']:
            print("\n💡 Suggestions:")
            for s in result['suggestions']:
                print(f"  - {s}")
        conv_manager.add_interaction(user_input, result['answer'], result['sources'])
        if conv_manager.feedback_due_now():
            fb = input("\n⭐ Please rate the last few answers (1-5) or leave feedback: ").strip()
            conv_manager.record_feedback({'feedback': fb, 'at': len(conv_manager.current_session)})
            print("Thank you for your feedback!")

if __name__ == '__main__':
    main() 