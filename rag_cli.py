"""
Command-line interface for the podcast RAG system.
Provides interactive querying and session management.
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.file_utils import get_file_utils
from rag_system.vector_database import VectorDatabase
from rag_system.query_processor import QueryProcessor
from pipeline.pipeline_runner import PipelineRunner, list_sessions

logger = get_logger(__name__)


class PodcastRAGCLI:
    """
    Command-line interface for the podcast RAG system
    """
    
    def __init__(self):
        self.vector_db = None
        self.query_processor = None
        self.current_session = None
        self.file_utils = get_file_utils()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            self.vector_db = VectorDatabase()
            self.query_processor = QueryProcessor(self.vector_db)
            logger.info("RAG components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            print(f"Error: {e}")
            print("Please ensure all dependencies are installed:")
            print("  pip install chromadb sentence-transformers")
    
    def run_interactive(self):
        """Run interactive query session"""
        print("\n" + "="*60)
        print("🎙️  PODCAST RAG SYSTEM - Interactive Mode")
        print("="*60)
        
        if not self.query_processor:
            print("❌ System not properly initialized. Please check dependencies.")
            return
        
        # Show available sessions
        self._show_session_selection()
        
        print("\n💡 Enter your queries or use commands:")
        print("   - 'help' for available commands")
        print("   - 'quit' or 'exit' to leave")
        print("   - 'sessions' to switch sessions")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                session_prompt = f"[{self.current_session[:8] if self.current_session else 'All'}] "
                query = input(f"\n{session_prompt}Query: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    self._show_help()
                elif query.lower() == 'sessions':
                    self._show_session_selection()
                elif query.lower() == 'stats':
                    self._show_database_stats()
                elif query.lower() == 'overview':
                    self._show_session_overview()
                elif query.lower() == 'topics':
                    self._show_popular_topics()
                elif query.startswith('session '):
                    session_id = query.split(' ', 1)[1]
                    self._set_current_session(session_id)
                else:
                    # Process query
                    self._process_query(query)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"CLI error: {e}")
    
    def _show_session_selection(self):
        """Show available sessions and allow selection"""
        print("\n📁 Available Sessions:")
        
        sessions = self.vector_db.list_sessions() if self.vector_db else []
        
        if not sessions:
            print("   No sessions found in database")
            print("   Use 'process <audio_file>' to add content")
            return
        
        print(f"   {'ID':<10} {'Date':<20} {'Duration':<10} {'Segments':<10}")
        print("   " + "-" * 50)
        
        for i, session in enumerate(sessions[:10], 1):  # Show top 10
            session_id = session['session_id'][:8] + "..." if len(session['session_id']) > 8 else session['session_id']
            date = session.get('processing_date', '')[:16] if session.get('processing_date') else 'Unknown'
            duration = f"{session.get('audio_duration', 0)/60:.1f}m"
            segments = str(session.get('total_segments', 0))
            
            print(f"   {session_id:<10} {date:<20} {duration:<10} {segments:<10}")
        
        print(f"\n   Found {len(sessions)} total sessions")
        print("   💡 Use 'session <session_id>' to filter queries to a specific session")
        print("   💡 Use 'sessions' to return to this menu")
    
    def _set_current_session(self, session_id: str):
        """Set current session filter"""
        # Find full session ID if partial provided
        sessions = self.vector_db.list_sessions() if self.vector_db else []
        
        matching_session = None
        for session in sessions:
            if session['session_id'].startswith(session_id):
                matching_session = session['session_id']
                break
        
        if matching_session:
            self.current_session = matching_session
            print(f"✅ Set session filter to: {matching_session[:16]}...")
        else:
            print(f"❌ Session not found: {session_id}")
    
    def _process_query(self, query: str):
        """Process and display query results"""
        print(f"\n🔍 Searching for: '{query}'")
        
        try:
            # Process query
            result = self.query_processor.process_query(query, self.current_session)
            
            # Display results
            self._display_query_results(result)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            logger.error(f"Query processing failed: {e}")
    
    def _display_query_results(self, result: Dict):
        """Display formatted query results"""
        response = result.get('response', {})
        results = result.get('results', [])
        metadata = result.get('metadata', {})
        
        # Show summary
        print(f"\n📊 {response.get('summary', 'No summary available')}")
        
        if not results:
            print("\n💡 Suggestions:")
            for suggestion in response.get('suggestions', []):
                print(f"   • {suggestion}")
            return
        
        # Show highlights
        if response.get('highlights'):
            print(f"\n🎯 Key Highlights:")
            for i, highlight in enumerate(response['highlights'], 1):
                print(f"\n   {i}. {highlight}")
        
        # Show detailed results
        print(f"\n📋 Detailed Results ({len(results)} found):")
        print("-" * 80)
        
        for i, res in enumerate(results[:5], 1):  # Show top 5
            doc = res['document']
            meta = res['metadata']
            similarity = res.get('similarity', 0)
            
            # Truncate long documents
            display_doc = doc[:200] + "..." if len(doc) > 200 else doc
            
            print(f"\n{i}. [Score: {similarity:.2f}] {display_doc}")
            
            # Show metadata
            meta_parts = []
            if 'speaker' in meta:
                meta_parts.append(f"Speaker: {meta['speaker']}")
            if 'time_range' in res:
                meta_parts.append(f"Time: {res['time_range']}")
            if 'emotion_label' in meta:
                meta_parts.append(f"Emotion: {meta['emotion_label']}")
            if 'type' in meta:
                meta_parts.append(f"Type: {meta['type']}")
            
            if meta_parts:
                print(f"   📝 {' | '.join(meta_parts)}")
        
        if len(results) > 5:
            print(f"\n   ... and {len(results) - 5} more results")
        
        # Show processing info
        processing_time = metadata.get('processing_time', 0)
        print(f"\n⏱️  Query processed in {processing_time:.2f}s")
        
        # Show suggestions
        if response.get('suggestions'):
            print(f"\n💡 Related suggestions:")
            for suggestion in response['suggestions']:
                print(f"   • {suggestion}")
    
    def _show_help(self):
        """Show help information"""
        print(f"\n📖 HELP - Available Commands:")
        print("-" * 40)
        print("🔍 SEARCH COMMANDS:")
        print("   <query>           - Search for content")
        print("   about <topic>     - Search by topic")
        print("   what did <name> say - Search by speaker")
        print("   <emotion> moments - Search by emotion")
        print("   summarize         - Get summary")
        print()
        print("📊 SESSION COMMANDS:")
        print("   sessions          - List all sessions")
        print("   session <id>      - Filter to specific session")
        print("   overview          - Show current session overview")
        print("   topics            - Show popular topics")
        print("   stats             - Show database statistics")
        print()
        print("🎛️  SYSTEM COMMANDS:")
        print("   help              - Show this help")
        print("   quit/exit         - Exit the program")
        print()
        print("💡 EXAMPLE QUERIES:")
        print("   'what did the host talk about'")
        print("   'summarize the main points'")
        print("   'happy moments in the conversation'")
        print("   'about machine learning'")
    
    def _show_database_stats(self):
        """Show database statistics"""
        if not self.vector_db:
            print("❌ Database not available")
            return
        
        print(f"\n📈 Database Statistics:")
        stats = self.vector_db.get_database_stats()
        
        if stats:
            print(f"   Total Embeddings: {stats.get('total_embeddings', 0):,}")
            print(f"   Total Sessions: {stats.get('total_sessions', 0)}")
            print(f"   Embedding Model: {stats.get('embedding_model', 'Unknown')}")
            print(f"   Database Path: {stats.get('database_path', 'Unknown')}")
            
            content_dist = stats.get('content_type_distribution', {})
            if content_dist:
                print(f"\n   Content Distribution:")
                for content_type, count in content_dist.items():
                    print(f"     {content_type}: {count:,}")
        else:
            print("   Unable to retrieve statistics")
    
    def _show_session_overview(self):
        """Show overview of current session"""
        if not self.current_session:
            print("❌ No session selected. Use 'session <id>' first.")
            return
        
        if not self.vector_db:
            print("❌ Database not available")
            return
        
        print(f"\n📊 Session Overview: {self.current_session[:16]}...")
        
        overview = self.vector_db.get_session_summary(self.current_session)
        
        if overview:
            print(f"   Total Duration: {overview.get('total_duration', 0)/60:.1f} minutes")
            print(f"   Total Embeddings: {overview.get('total_embeddings', 0):,}")
            print(f"   Speakers: {', '.join(overview.get('speakers', []))}")
            
            if overview.get('overall_summary'):
                summary = overview['overall_summary']
                display_summary = summary[:300] + "..." if len(summary) > 300 else summary
                print(f"\n   Summary: {display_summary}")
            
            content_counts = overview.get('content_counts', {})
            if content_counts:
                print(f"\n   Content Types:")
                for content_type, count in content_counts.items():
                    print(f"     {content_type}: {count}")
        else:
            print("   Unable to retrieve session overview")
    
    def _show_popular_topics(self):
        """Show popular topics in current session or globally"""
        if not self.query_processor:
            print("❌ Query processor not available")
            return
        
        print(f"\n🏷️  Popular Topics:")
        
        try:
            topics = self.query_processor.get_popular_topics(
                session_id=self.current_session,
                limit=10
            )
            
            if topics:
                for i, topic_info in enumerate(topics, 1):
                    topic = topic_info['topic']
                    count = topic_info['count']
                    print(f"   {i}. {topic} ({count} mentions)")
            else:
                print("   No topics found")
                
        except Exception as e:
            print(f"   Error retrieving topics: {e}")
    
    def process_audio_file(self, audio_file: str, session_id: Optional[str] = None):
        """Process audio file through pipeline and add to RAG system"""
        print(f"\n🎵 Processing audio file: {audio_file}")
        
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            return
        
        try:
            # Run pipeline
            runner = PipelineRunner(session_id=session_id)
            results = runner.process_audio_file(audio_file)
            
            print(f"✅ Pipeline completed successfully")
            
            # Add to RAG system
            if self.vector_db:
                session_id = self.vector_db.add_podcast_session(results)
                print(f"✅ Added to RAG system | Session ID: {session_id}")
                self.current_session = session_id
            else:
                print("⚠️  RAG system not available - results saved but not indexed")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            logger.error(f"Audio processing failed: {e}")
    
    def list_pipeline_sessions(self):
        """List sessions from pipeline output directory"""
        print(f"\n📁 Pipeline Sessions:")
        
        try:
            sessions = list_sessions()
            
            if not sessions:
                print("   No sessions found")
                return
            
            print(f"   {'Session ID':<20} {'Audio File':<30} {'Status':<15}")
            print("   " + "-" * 65)
            
            for session in sessions[:10]:  # Show top 10
                session_id = session['session_id'][:18] + "..." if len(session['session_id']) > 18 else session['session_id']
                audio_file = os.path.basename(session.get('audio_file', 'Unknown'))[:28]
                status = session.get('last_completed_step', 'Unknown')[:13]
                
                print(f"   {session_id:<20} {audio_file:<30} {status:<15}")
            
            print(f"\n   Found {len(sessions)} total sessions")
            
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Podcast RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python rag_cli.py
  
  # Process audio file
  python rag_cli.py --process audio.mp3
  
  # Single query
  python rag_cli.py --query "what did they discuss about AI"
  
  # List sessions
  python rag_cli.py --list-sessions
        """
    )
    
    parser.add_argument(
        '--process',
        type=str,
        help='Process audio file through pipeline and add to RAG system'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Execute single query and exit'
    )
    
    parser.add_argument(
        '--session',
        type=str,
        help='Session ID to filter queries (use with --query)'
    )
    
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List all available sessions'
    )
    
    parser.add_argument(
        '--list-pipeline',
        action='store_true',
        help='List pipeline sessions'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = PodcastRAGCLI()
    
    try:
        if args.process:
            cli.process_audio_file(args.process)
        elif args.query:
            cli.current_session = args.session
            cli._process_query(args.query)
        elif args.list_sessions:
            cli._show_session_selection()
        elif args.list_pipeline:
            cli.list_pipeline_sessions()
        elif args.stats:
            cli._show_database_stats()
        else:
            # Interactive mode
            cli.run_interactive()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"CLI error: {e}")


if __name__ == "__main__":
    main()
