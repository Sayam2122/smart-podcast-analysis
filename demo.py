"""
Demonstration script for the podcast analysis pipeline with RAG system.
Shows complete workflow from audio processing to interactive querying.
"""

import os
import sys
from pathlib import Path
import time
import json
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from pipeline.pipeline_runner import process_podcast
from rag_system.vector_database import VectorDatabase
from rag_system.query_processor import QueryProcessor

logger = get_logger(__name__)


def main():
    """
    Demonstration of the complete podcast analysis and RAG system
    """
    print("🎙️ PODCAST ANALYSIS PIPELINE WITH RAG SYSTEM")
    print("=" * 60)
    
    # Check if we have a sample audio file
    sample_audio = find_sample_audio()
    
    if sample_audio:
        print(f"📁 Found sample audio: {sample_audio}")
        
        # Option 1: Process new audio
        choice = input("\n1. Process sample audio\n2. Use existing results\n3. Demo RAG queries only\nChoice (1-3): ").strip()
        
        if choice == "1":
            demo_complete_pipeline(sample_audio)
        elif choice == "2":
            demo_with_existing_results()
        else:
            demo_rag_queries_only()
    else:
        print("📁 No sample audio found")
        print("💡 Please place an audio file (MP3, WAV, M4A, FLAC) in the project directory")
        print("💡 Or run: python demo.py --help for more options")
        
        # Demo RAG queries only
        demo_rag_queries_only()


def find_sample_audio():
    """Find sample audio file in project directory"""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    
    for file_path in Path('.').glob('*'):
        if file_path.suffix.lower() in audio_extensions:
            return str(file_path)
    
    return None


def demo_complete_pipeline(audio_file: str):
    """
    Demonstrate complete pipeline: Audio processing → RAG system → Queries
    """
    print(f"\n🚀 STARTING COMPLETE PIPELINE DEMO")
    print("-" * 40)
    
    try:
        # Step 1: Process audio through pipeline
        print("📊 Step 1: Processing audio through pipeline...")
        start_time = time.time()
        
        results = process_podcast(
            audio_file_path=audio_file,
            resume=True  # Resume if already partially processed
        )
        
        processing_time = time.time() - start_time
        print(f"✅ Pipeline completed in {processing_time:.1f}s")
        
        # Show basic results
        show_pipeline_results(results)
        
        # Step 2: Add to RAG system
        print("\n📚 Step 2: Adding results to RAG system...")
        
        vector_db = VectorDatabase()
        session_id = vector_db.add_podcast_session(results)
        
        print(f"✅ Added to RAG database | Session ID: {session_id}")
        
        # Step 3: Demo interactive queries
        print("\n🔍 Step 3: Interactive querying...")
        demo_interactive_queries(vector_db, session_id)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")


def demo_with_existing_results():
    """
    Demo using existing pipeline results
    """
    print(f"\n📂 DEMO WITH EXISTING RESULTS")
    print("-" * 40)
    
    # Look for existing session results
    output_dir = Path("output/sessions")
    
    if not output_dir.exists():
        print("❌ No existing results found")
        print("💡 Run pipeline first with: python demo.py")
        return
    
    # Find most recent session
    session_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not session_dirs:
        print("❌ No session directories found")
        return
    
    # Get most recent
    latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_session / "complete_results.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    try:
        # Load results
        print(f"📁 Loading results from: {latest_session.name}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        show_pipeline_results(results)
        
        # Add to RAG system
        print("\n📚 Adding to RAG system...")
        
        vector_db = VectorDatabase()
        session_id = vector_db.add_podcast_session(results)
        
        print(f"✅ Added to RAG database | Session ID: {session_id}")
        
        # Demo queries
        demo_interactive_queries(vector_db, session_id)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")


def demo_rag_queries_only():
    """
    Demo RAG queries using existing database content
    """
    print(f"\n🔍 RAG QUERIES DEMO")
    print("-" * 40)
    
    try:
        # Initialize RAG system
        vector_db = VectorDatabase()
        query_processor = QueryProcessor(vector_db)
        
        # Check if we have any content
        stats = vector_db.get_database_stats()
        
        if stats.get('total_embeddings', 0) == 0:
            print("❌ No content in RAG database")
            print("💡 Process audio first or import existing results")
            return
        
        print(f"📊 Database contains {stats.get('total_embeddings', 0)} embeddings")
        print(f"📁 Across {stats.get('total_sessions', 0)} sessions")
        
        # Demo queries
        demo_interactive_queries(vector_db)
        
    except Exception as e:
        print(f"❌ RAG demo failed: {e}")
        logger.error(f"RAG demo error: {e}")


def show_pipeline_results(results: Dict):
    """Show summary of pipeline results"""
    print("\n📊 PIPELINE RESULTS SUMMARY:")
    
    # Audio info
    if 'audio_data' in results:
        audio_info = results['audio_data']
        print(f"   🎵 Audio Duration: {audio_info.get('duration', 0):.1f}s")
        print(f"   🔊 Sample Rate: {audio_info.get('sample_rate', 0)} Hz")
    
    # Transcription
    if 'transcription' in results:
        segments = results['transcription'].get('segments', [])
        print(f"   📝 Transcription: {len(segments)} segments")
    
    # Speakers
    if 'diarization' in results:
        speakers = set()
        for seg in results['diarization'].get('segments', []):
            speakers.add(seg.get('speaker', 'Unknown'))
        print(f"   👥 Speakers: {len(speakers)} detected")
    
    # Emotions
    if 'emotion_analysis' in results:
        emotions = {}
        for seg in results['emotion_analysis']:
            emotion = seg.get('emotion', {}).get('label', 'unknown')
            emotions[emotion] = emotions.get(emotion, 0) + 1
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'unknown'
        print(f"   😊 Emotions: {dominant_emotion} (dominant)")
    
    # Semantic blocks
    if 'semantic_blocks' in results:
        blocks = results['semantic_blocks']
        print(f"   📚 Semantic Blocks: {len(blocks)}")
    
    # Summaries
    if 'summaries' in results:
        summaries = results['summaries']
        print(f"   📋 Summaries: {len(summaries)} blocks summarized")


def demo_interactive_queries(vector_db: VectorDatabase, session_id: str = None):
    """Demo interactive queries"""
    query_processor = QueryProcessor(vector_db)
    
    # Predefined demo queries
    demo_queries = [
        "summarize the main points",
        "what are the key topics discussed",
        "who said the most",
        "any emotional moments",
        "what was discussed about technology"
    ]
    
    print(f"\n🎯 DEMO QUERIES:")
    print("Choose a query to run:")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"   {i}. {query}")
    
    print(f"   {len(demo_queries) + 1}. Custom query")
    print(f"   {len(demo_queries) + 2}. Interactive mode")
    
    try:
        choice = input(f"\nChoice (1-{len(demo_queries) + 2}): ").strip()
        
        if choice.isdigit():
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(demo_queries):
                # Run predefined query
                query = demo_queries[choice_num - 1]
                run_demo_query(query_processor, query, session_id)
                
            elif choice_num == len(demo_queries) + 1:
                # Custom query
                custom_query = input("Enter your query: ").strip()
                if custom_query:
                    run_demo_query(query_processor, custom_query, session_id)
                
            elif choice_num == len(demo_queries) + 2:
                # Interactive mode
                run_interactive_queries(query_processor, session_id)
        
    except ValueError:
        print("❌ Invalid choice")
    except KeyboardInterrupt:
        print("\n👋 Demo ended")


def run_demo_query(query_processor: QueryProcessor, query: str, session_id: str = None):
    """Run a single demo query"""
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    try:
        start_time = time.time()
        result = query_processor.process_query(query, session_id)
        query_time = time.time() - start_time
        
        # Show results
        response = result.get('response', {})
        results = result.get('results', [])
        
        print(f"📊 {response.get('summary', 'No summary')}")
        print(f"⏱️  Query time: {query_time:.2f}s")
        print(f"📋 Results found: {len(results)}")
        
        # Show top results
        if results:
            print(f"\n🎯 Top Results:")
            
            for i, res in enumerate(results[:3], 1):
                doc = res['document']
                metadata = res['metadata']
                similarity = res.get('similarity', 0)
                
                # Truncate long documents
                display_doc = doc[:150] + "..." if len(doc) > 150 else doc
                
                print(f"\n   {i}. [Score: {similarity:.2f}] {display_doc}")
                
                # Show relevant metadata
                meta_parts = []
                if 'speaker' in metadata:
                    meta_parts.append(f"Speaker: {metadata['speaker']}")
                if 'time_range' in res:
                    meta_parts.append(f"Time: {res['time_range']}")
                if 'emotion_label' in metadata:
                    meta_parts.append(f"Emotion: {metadata['emotion_label']}")
                
                if meta_parts:
                    print(f"      📝 {' | '.join(meta_parts)}")
        
        # Show suggestions
        if response.get('suggestions'):
            print(f"\n💡 Suggestions:")
            for suggestion in response['suggestions']:
                print(f"   • {suggestion}")
    
    except Exception as e:
        print(f"❌ Query failed: {e}")


def run_interactive_queries(query_processor: QueryProcessor, session_id: str = None):
    """Run interactive query loop"""
    print(f"\n🔄 INTERACTIVE QUERY MODE")
    print("=" * 40)
    print("Enter queries or 'quit' to exit")
    
    session_filter = f" [Session: {session_id[:8]}...]" if session_id else " [All Sessions]"
    print(f"Context:{session_filter}")
    print("-" * 40)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive mode")
                break
            
            run_demo_query(query_processor, query, session_id)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Podcast Analysis Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Interactive demo
  python demo.py --audio file.mp3  # Process specific audio file
  python demo.py --rag-only        # Demo RAG queries only
        """
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        help='Audio file to process'
    )
    
    parser.add_argument(
        '--rag-only',
        action='store_true',
        help='Demo RAG queries only (skip audio processing)'
    )
    
    args = parser.parse_args()
    
    if args.audio:
        if os.path.exists(args.audio):
            demo_complete_pipeline(args.audio)
        else:
            print(f"❌ Audio file not found: {args.audio}")
    elif args.rag_only:
        demo_rag_queries_only()
    else:
        main()
