# /podcast_rag_project/main.py

import logging
import os
import sys
from config import (
    EPISODES_DIR, REQUIRED_FILES, VECTOR_STORE_DIR,
    EMBEDDING_MODEL, OLLAMA_MODEL, CONVERSATION_HISTORY_DIR,
    CONTENT_ASSETS_DIR, ENABLE_ENHANCED_CHUNKING, ENABLE_CONTENT_EXTRACTION
)
from data_loader import DataLoader
from vector_store import VectorStore
# Lazy import for heavy libraries to avoid memory issues
# from enhanced_chunking import enhanced_contextual_chunking
# from chunking import group_segments_into_chunks
from enhanced_query_engine import EnhancedQueryEngine
from conversation_manager import ConversationManager
from content_extractor import ContentExtractor

# --- Setup Logging ---
# A centralized logging setup for clear and informative console output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_available_episodes():
    """
    Scans the episodes directory and returns a list of available episode IDs.
    An episode is considered "available" if it's a directory within EPISODES_DIR.
    """
    if not os.path.isdir(EPISODES_DIR):
        logging.error(f"Episodes directory not found at: {EPISODES_DIR}")
        logging.error("Please create it and place your episode folders inside.")
        return []
    return [d for d in os.listdir(EPISODES_DIR) if os.path.isdir(os.path.join(EPISODES_DIR, d))]

def initialize_engine_for_episode(episode_id):
    """
    Initializes all necessary components (DataLoader, VectorStore, QueryEngine)
    for a single, specific episode. This is the core setup function.
    """
    logging.info(f"Initializing enhanced query engine for episode: {episode_id}")
    
    # 1. Initialize VectorStore for the selected episode.
    #    This will automatically check if a pre-built index exists on disk.
    vector_store = VectorStore(VECTOR_STORE_DIR, episode_id, EMBEDDING_MODEL)

    # 2. If the vector store isn't built, build it now.
    #    This is a one-time process per episode.
    if not vector_store.is_built():
        logging.warning(f"Vector store for '{episode_id}' not found. Building now...")
        loader = DataLoader(EPISODES_DIR, REQUIRED_FILES)
        merged_data = loader.load_and_merge_data(episode_id)

        if not merged_data:
            logging.error(f"Failed to load data for '{episode_id}'. Cannot build vector store.")
            return None, None, None

        # Apply enhanced chunking strategy for better contextual understanding
        if ENABLE_ENHANCED_CHUNKING:
            try:
                from sentence_transformers import SentenceTransformer
                from enhanced_chunking import enhanced_contextual_chunking
                embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                chunked_data = enhanced_contextual_chunking(merged_data, embedding_model)
                logging.info("Applied enhanced contextual chunking")
            except (ImportError, MemoryError) as e:
                logging.warning(f"Enhanced chunking failed ({e}), using standard chunking")
                from chunking import group_segments_into_chunks
                chunked_data = group_segments_into_chunks(merged_data)
                logging.info("Applied standard chunking")
        else:
            from chunking import group_segments_into_chunks
            chunked_data = group_segments_into_chunks(merged_data)
            logging.info("Applied standard chunking")

        vector_store.build(chunked_data)
        logging.info(f"Successfully built and saved vector store for '{episode_id}'.")
    
    # 3. Initialize conversation manager
    conversation_manager = ConversationManager(CONVERSATION_HISTORY_DIR)
    
    # 4. Initialize content extractor if enabled
    content_extractor = None
    if ENABLE_CONTENT_EXTRACTION:
        content_extractor = ContentExtractor(OLLAMA_MODEL)
    
    # 5. Initialize the Enhanced QueryEngine with all components
    query_engine = EnhancedQueryEngine(
        vector_store, 
        OLLAMA_MODEL, 
        conversation_manager,
        content_extractor
    )
    
    return query_engine, conversation_manager, content_extractor

def main():
    """
    Runs the enhanced interactive loop for the multi-episode RAG system.
    This function manages the user interface, episode selection, and command handling.
    """
    print("=== ENHANCED PODCAST RAG SYSTEM ===")
    print("🚀 Advanced AI-Powered Podcast Analysis & Content Extraction")
    print("Features: Contextual Memory • Smart Chunking • Content Assets • Grounded Responses")
    print()

    # Ensure the necessary directories exist before starting.
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(EPISODES_DIR, exist_ok=True)
    os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
    os.makedirs(CONTENT_ASSETS_DIR, exist_ok=True)
    
    engine = None
    conversation_manager = None
    content_extractor = None
    current_episode = None
    session_id = None

    while True:
        # --- Episode Selection Menu ---
        if not engine:
            print("\n=== EPISODE SELECTION ===")
            available_episodes = list_available_episodes()
            if not available_episodes:
                print(f"\n❌ ERROR: No episodes found in '{EPISODES_DIR}'.")
                print("To get started, please follow these steps:")
                print("1. Create a folder for your episode inside the 'data/episodes/' directory (e.g., 'data/episodes/my_first_episode/').")
                print("2. Place all the required JSON files for that episode inside its folder.")
                print("3. Rerun this script.")
                sys.exit(1)
            
            print("📁 Available episodes:")
            for i, ep_id in enumerate(available_episodes):
                print(f"  {i+1}. {ep_id}")
            
            choice = input(f"\n📝 Select episode (1-{len(available_episodes)}) or 'quit': ").strip()
            
            if choice.lower() == 'quit':
                break
            
            try:
                episode_index = int(choice) - 1
                if 0 <= episode_index < len(available_episodes):
                    current_episode = available_episodes[episode_index]
                    print(f"\n🎯 Initializing episode: {current_episode}")
                    components = initialize_engine_for_episode(current_episode)
                    if components and len(components) == 3:
                        engine, conversation_manager, content_extractor = components
                        if conversation_manager:
                            session_id = conversation_manager.start_new_session(current_episode)
                            print(f"✅ Started session: {session_id}")
                        print(f"🚀 Ready to analyze episode: {current_episode}")
                    else:
                        print("❌ Failed to initialize episode.")
                        continue
                else:
                    print("❌ Invalid selection. Please try again.")
                    continue
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'quit'.")
                continue

        # --- Main Query Interface ---
        print(f"\n=== 🎙️ ANALYZING EPISODE: {current_episode} ===")
        print("💡 Commands: 'extract', 'analyze', 'suggest', 'reload', 'reindex', 'switch', 'quit'")
        
        user_input = input("\n🤔 Your Question: ").strip()

        if user_input.lower() == 'quit':
            if engine:
                engine.save_session()
                print("💾 Session saved successfully!")
            break

        if user_input.lower() == 'switch':
            if engine:
                engine.save_session()
                print("💾 Session saved!")
            engine = None
            conversation_manager = None
            content_extractor = None
            current_episode = None
            session_id = None
            continue

        if user_input.lower() == 'reload':
            print("\n🔄 Reloading episode data and conversation...")
            components = initialize_engine_for_episode(current_episode)
            if components and len(components) == 3:
                engine, conversation_manager, content_extractor = components
                if conversation_manager:
                    session_id = conversation_manager.start_new_session(current_episode)
                    print(f"✅ Reloaded and started new session: {session_id}")
                else:
                    print("✅ Reloaded episode data.")
            else:
                print("❌ Failed to reload episode data.")
            continue

        if user_input.lower() == 'reindex':
            print("\n🔄 Rebuilding vector store and index for this episode...")
            loader = DataLoader(EPISODES_DIR, REQUIRED_FILES)
            merged_data = loader.load_and_merge_data(current_episode)
            if not merged_data:
                print("❌ Failed to reload episode data for reindexing.")
                continue
            if ENABLE_ENHANCED_CHUNKING:
                try:
                    from sentence_transformers import SentenceTransformer
                    from enhanced_chunking import enhanced_contextual_chunking
                    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                    chunked_data = enhanced_contextual_chunking(merged_data, embedding_model)
                except (ImportError, MemoryError) as e:
                    logging.warning(f"Enhanced chunking failed ({e}), using standard chunking")
                    from chunking import group_segments_into_chunks
                    chunked_data = group_segments_into_chunks(merged_data)
            else:
                from chunking import group_segments_into_chunks
                chunked_data = group_segments_into_chunks(merged_data)
            vector_store = VectorStore(VECTOR_STORE_DIR, current_episode, EMBEDDING_MODEL)
            vector_store.build(chunked_data)
            print("✅ Vector store and index rebuilt successfully.")
            continue

        if user_input.lower() == 'extract':
            print("\n🔄 Extracting content assets...")
            if content_extractor:
                assets = engine.extract_episode_content(current_episode)
                if 'error' not in assets:
                    print("\n✅ CONTENT EXTRACTION COMPLETE!")
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
                    print(f"\n💾 Full assets saved to content_assets/{current_episode}_content_assets.json")
                else:
                    print(f"❌ Extraction failed: {assets.get('error', 'Unknown error')}")
            else:
                print("❌ Content extraction not available")
            continue

        if user_input.lower() == 'analyze':
            if conversation_manager:
                analysis = engine.get_conversation_analysis()
                if analysis:
                    print("\n📊 CONVERSATION ANALYSIS:")
                    print(f"  • Total interactions: {analysis.get('total_interactions', 0)}")
                    print(f"  • Session duration: {analysis.get('session_duration_minutes', 0)} minutes")
                    print(f"  • Engagement level: {analysis.get('engagement_level', 'unknown')}")
                    print(f"  • Question types: {analysis.get('question_types', {})}")
                    print(f"  • Topic evolution: {' → '.join(analysis.get('topic_evolution', [])[:5])}")
                else:
                    print("❌ No conversation data to analyze yet")
            else:
                print("❌ Conversation tracking not available")
            continue

        if user_input.lower() == 'suggest':
            print("\n💡 INTELLIGENT SUGGESTIONS:")
            if engine and conversation_manager and conversation_manager.current_session:
                last_interaction = conversation_manager.current_session[-1]
                suggestions = engine.generate_followup_suggestions(
                    last_interaction['ai_response'], 
                    last_interaction['sources_used']
                )
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

        if not user_input:
            print("❌ Please enter a question or command.")
            continue

        print("\n🔄 Processing your question...")
        answer, sources = engine.ask(user_input)
        print("\n" + "="*60)
        print("🎯 CONFIDENT ANALYSIS")
        print("="*60)
        print(answer)
        if sources:
            print(f"\n📍 SOURCES USED ({len(sources)} segments):")
            for i, source in enumerate(sources[:3], 1):
                start_time = source.get('start_time', 0)
                end_time = source.get('end_time', 0)
                speaker = source.get('primary_speaker', source.get('speaker', 'Unknown'))
                emotion = source.get('dominant_emotion', source.get('text_emotion', {}).get('emotion', 'neutral'))
                print(f"\n  📍 Source {i}:")
                print(f"     ⏰ Time: {start_time:.1f}s - {end_time:.1f}s")
                print(f"     🎤 Speaker: {speaker}")
                print(f"     😊 Tone: {emotion}")
                print(f"     💬 Content: \"{source.get('text', '')[:100]}...\"")
        suggestions = engine.generate_followup_suggestions(answer, sources)
        if suggestions:
            print(f"\n💡 FOLLOW-UP SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
