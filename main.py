"""
Smart Audio RAG System - Main Entry Point

This is the main entry point for the Smart Audio RAG System.
Run this file to start the interactive CLI interface.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the Smart Audio RAG System."""
    parser = argparse.ArgumentParser(description="Smart Audio RAG System")
    parser.add_argument("mode", nargs="?", default="interactive", 
                       choices=["interactive", "query"], 
                       help="Mode to run: interactive or query")
    parser.add_argument("--query", "-q", help="Query to execute (for query mode)")
    parser.add_argument("--data-dir", "-d", default="output/sessions", 
                       help="Directory containing session data")
    
    args = parser.parse_args()
    
    try:
        from rag_cli import PodcastRAGCLI
        
        # Set up environment
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Avoid tokenizer warnings
        
        # Initialize CLI
        cli = PodcastRAGCLI()
        
        if args.mode == "interactive":
            print("🎙️ Smart Audio RAG System - Interactive Mode")
            cli.interactive_mode()
        elif args.mode == "query" and args.query:
            # Initialize the system first
            print("🎙️ Initializing RAG System...")
            success = cli.initialize_system(chroma_db_path="chroma_db")
            if success:
                # Execute query using query engine
                results = cli.query_engine.query(
                    query_text=args.query,
                    session_filter=None,
                    include_context=True,
                    show_complete_content=True
                )
                cli._print_query_results(results)
            else:
                print("❌ Failed to initialize RAG system")
                sys.exit(1)
        else:
            print("❌ Invalid arguments. Use --help for usage information.")
            sys.exit(1)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting Smart Audio RAG System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
