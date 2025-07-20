import os
import sys
from data_loader import DataLoader
from vector_store import VectorStore
from query_engine import QueryEngine
from conversation_manager import ConversationManager
from analytics import Analytics
from content_generator import ContentGenerator
import re

# --- Config ---
# Search for episodes in the output directory in the parent folder of the current workspace
EPISODES_DIR = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 'output', 'sessions')
VECTOR_STORE_DIR = 'vector_store'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
OLLAMA_MODEL = 'mistral:latest'


def list_episodes():
    if not os.path.isdir(EPISODES_DIR):
        print(f"[main] Episodes directory not found: {EPISODES_DIR}")
        return []
    return [d for d in os.listdir(EPISODES_DIR) if os.path.isdir(os.path.join(EPISODES_DIR, d))]


def build_vector_store_for_episode(episode_id, loader):
    segments = loader.load_episode(episode_id)
    if not segments:
        print(f"[main] No data for episode: {episode_id}")
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
    if any(word in q for word in ['social', 'tweet', 'post', 'thread', 'card']):
        return 'assets'
    if any(word in q for word in ['audio', 'mp3', 'wav', 'make audio', 'save audio']):
        return 'audio'
    return 'standard'


def extract_quotes(segments):
    # Extract segments that look like quotes or shlokas (heuristic: contains Sanskrit chars or is marked as a quote)
    quotes = []
    for seg in segments:
        text = seg.get('text', '')
        if re.search(r'[\u0900-\u097F]', text) or 'quote' in text.lower() or 'shloka' in text.lower():
            quotes.append({
                'text': text,
                'timestamp': f"{seg.get('start_time', 0):.1f}s",
                'speaker': seg.get('speaker', 'Unknown'),
                'block_id': seg.get('block_id'),
                'segment_id': seg.get('segment_id')
            })
    return quotes


def detect_domain_and_guidance(episode_model, llm_client):
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
    print("\n=== Podcast RAG CLI ===")
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
    domain, summary, auto_guidance = detect_domain_and_guidance(merged_episode_model, query_engine.llm_client)
    user_guidance = ''
    print(f"\n[System] Detected Domain: {domain}")
    print(f"[System] Domain Summary: {summary}")
    print(f"[System] Guidance: {auto_guidance}")
    print(f"\nCLI session started: {session_id}")
    print("\n[Optional] You can provide your own domain guidance (e.g. 'This is a spiritual podcast, focus on philosophy and emotional tone.')")
    print("Type 'guidance' to set or update domain guidance at any time.")

    # --- Main Loop ---
    while True:
        user_input = input("\n💬 You: ").strip()
        if user_input.lower() in {'quit', 'exit'}:
            conv_manager.save_session()
            print("Session ended. Goodbye!")
            break
        if user_input.lower() == 'help':
            print("""
Commands:
  [question]         Ask anything about the episode(s)
  analytics          Show advanced analytics
  posts              Generate social media posts
  quotes             Generate quote cards
  thread             Generate a thread
  extract            Extract content assets
  suggest            Get follow-up suggestions
  reload             Reload vector store and episode data
  reindex            Rebuild vector store and index for this episode
  load [session_id]  Load previous session
  save               Save current session
  switch             Switch to another episode
  guidance           Set or update domain guidance
  new                Start a new session
  quit/exit          Exit and save this session's history
""")
            continue
        if user_input.lower() == 'guidance':
            print("\nYou can provide your own domain guidance (e.g. 'This is a spiritual podcast, focus on philosophy and emotional tone.')")
            user_guidance = input("Enter your domain guidance: ").strip()
            if user_guidance:
                print(f"[System] User guidance set: {user_guidance}")
            else:
                print("[System] User guidance cleared. Using auto-detected guidance only.")
            continue
        if user_input.lower() == 'analytics':
            print("\n=== Analytics ===")
            for eid in selected_episodes:
                print(f"--- {eid} ---")
                print("Speaker Dynamics:", Analytics.speaker_dynamics(segments_map[eid]))
                print("Emotional Patterns:", Analytics.emotional_patterns(segments_map[eid]))
                print("Content Metrics:", Analytics.content_metrics(segments_map[eid]))
                print("Topic Evolution:", Analytics.topic_evolution(segments_map[eid]))
            continue
        if user_input.lower() == 'posts':
            print("\n=== Social Media Posts ===")
            for eid in selected_episodes:
                print(f"--- {eid} ---")
                posts = ContentGenerator.social_media_posts(segments_map[eid])
                for p in posts:
                    print(p)
            continue
        if user_input.lower() == 'quotes':
            print("\n=== Quote Cards ===")
            for eid in selected_episodes:
                print(f"--- {eid} ---")
                cards = ContentGenerator.quote_cards(segments_map[eid])
                for c in cards:
                    print(c)
            continue
        if user_input.lower() == 'thread':
            print("\n=== Thread ===")
            for eid in selected_episodes:
                print(f"--- {eid} ---")
                thread = ContentGenerator.thread_generation(segments_map[eid])
                for t in thread:
                    print(t)
            continue
        if user_input.lower() == 'extract':
            print("\n🔄 Extracting content assets...")
            for eid in selected_episodes:
                assets = query_engine.extract_episode_content(eid)
                if 'error' not in assets:
                    print(f"\n✅ CONTENT EXTRACTION COMPLETE for {eid}!")
                    if 'quotes' in assets and assets['quotes']:
                        print(f"\n📝 KEY QUOTES ({len(assets['quotes'])} found):")
                        for i, quote in enumerate(assets['quotes'][:3], 1):
                            print(f"  {i}. \"{quote.get('text', '')}\"")
                            print(f"     Theme: {quote.get('theme', 'N/A')} | Confidence: {quote.get('confidence', 0)}/10")
                    if 'social_assets' in assets and assets['social_assets']:
                        social = assets['social_assets']
                        if social.get('summary'):
                            print(f"\n📱 EPISODE SUMMARY:\n{social['summary']}")
                        if social.get('taglines'):
                            print(f"\n🏷️ TAGLINES: {', '.join(social['taglines'][:3])}")
                    print(f"\n💾 Full assets saved to content_assets/{eid}_content_assets.json")
                else:
                    print(f"❌ Extraction failed for {eid}: {assets.get('error', 'Unknown error')}")
            continue
        if user_input.lower() == 'suggest':
            print("\n💡 INTELLIGENT SUGGESTIONS:")
            if conv_manager.current_session:
                last_interaction = conv_manager.current_session[-1]
                suggestions = query_engine._suggest_followups(last_interaction['user'], last_interaction['ai'], last_interaction['sources'])
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
            else:
                default_suggestions = [
                    "What are the main themes discussed in this episode?",
                    "Can you extract key quotes from the conversation?",
                    "What insights can you provide about the speakers?",
                    "How would you summarize the most important points?"
                ]
                for i, suggestion in enumerate(default_suggestions, 1):
                    print(f"  {i}. {suggestion}")
            continue
        if user_input.lower().startswith('load '):
            _, sid = user_input.split(maxsplit=1)
            if conv_manager.load_session(sid):
                print(f"Loaded session: {sid}")
            else:
                print(f"Failed to load session: {sid}")
            continue
        if user_input.lower() == 'save':
            conv_manager.save_session()
            print("Session saved.")
            continue
        if user_input.lower() == 'switch':
            print("\nSwitching to another episode...")
            conv_manager.save_session()
            print("Session saved!")
            # Reset state for new episode selection
            return main()
        if user_input.lower() == 'reindex':
            print("\n🔄 Rebuilding vector store and index for this episode...")
            for eid in selected_episodes:
                loader = DataLoader(EPISODES_DIR)
                merged_data = loader.load_episode(eid)
                if not merged_data:
                    print(f"❌ Failed to reload episode data for reindexing: {eid}")
                    continue
                from chunking import group_segments_into_chunks
                chunked_data = group_segments_into_chunks(merged_data)
                vector_store = VectorStore(VECTOR_STORE_DIR, eid, EMBEDDING_MODEL)
                vector_store.build(chunked_data)
                print(f"✅ Vector store and index rebuilt successfully for {eid}.")
            continue
        if user_input.lower() == 'new':
            print("\nStarting a new session...")
            conv_manager = ConversationManager()
            session_id = conv_manager.start_new_session("_".join(selected_episodes))
            # Redetect domain/guidance for new session
            domain, summary, guidance = detect_domain_and_guidance(merged_episode_model, query_engine.llm_client)
            print(f"\n[System] Detected Domain: {domain}")
            print(f"[System] Domain Summary: {summary}")
            print(f"[System] Guidance: {guidance}")
            print(f"New session started: {session_id}")
            continue
        if user_input.lower() == 'reload':
            print("Reloading vector store and episode data...")
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
            multi_vector_store = MultiVectorStore(vector_stores)
            merged_episode_model = merge_episode_models(episode_models)
            all_segments = []
            for segs in segments_map.values():
                all_segments.extend(segs)
            query_engine = QueryEngine(multi_vector_store, OLLAMA_MODEL, merged_episode_model, all_segments)
            # Redetect domain/guidance after reload
            domain, summary, guidance = detect_domain_and_guidance(merged_episode_model, query_engine.llm_client)
            print(f"\n[System] Detected Domain: {domain}")
            print(f"[System] Domain Summary: {summary}")
            print(f"[System] Guidance: {guidance}")
            continue
        intent = detect_intent(user_input)
        # When building the prompt, combine user_guidance and auto_guidance
        effective_guidance = user_guidance if user_guidance else auto_guidance
        query_engine.domain_guidance = effective_guidance
        # Optimize conversation history: only last 2-3 turns
        if hasattr(query_engine, 'conversation_history'):
            query_engine.conversation_history = query_engine.conversation_history[-3:]
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