"""
Main Entry Point for Audio RAG System

Provides command-line interface for launching the RAG system with various options.
Supports session management, batch processing, and interactive mode.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import system components
try:
    from session_manager import SessionManager
    from rag_engine import RAGEngine
    from enhanced_cli import AudioRAGCLI
    from report_generator import ReportGenerator
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are installed and in the correct directory.")
    SessionManager = None
    RAGEngine = None
    AudioRAGCLI = None
    ReportGenerator = None
    IMPORTS_AVAILABLE = False

def setup_argument_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Audio RAG System - Query and analyze audio transcriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive CLI mode
  python main.py --load-all               # Load all sessions and start CLI
  python main.py --session session1      # Load specific session
  python main.py --query "What was discussed about AI?"
  python main.py --report summary        # Generate summary report
  python main.py --sessions              # List available sessions
        """
    )
    
    # Session management
    parser.add_argument(
        "--sessions", 
        action="store_true",
        help="List available sessions and exit"
    )
    
    parser.add_argument(
        "--session", 
        action="append",
        help="Load specific session(s) (can be used multiple times)"
    )
    
    parser.add_argument(
        "--load-all", 
        action="store_true",
        help="Load all available sessions"
    )
    
    parser.add_argument(
        "--pattern",
        help="Select sessions matching pattern"
    )
    
    # Query operations
    parser.add_argument(
        "--query", 
        help="Perform a single query and exit"
    )
    
    parser.add_argument(
        "--smart-query", 
        help="Perform LLM-enhanced query and exit"
    )
    
    parser.add_argument(
        "--search",
        help="Search with specific criteria (e.g., 'speaker:John text:meeting')"
    )
    
    # Report generation
    parser.add_argument(
        "--report",
        choices=["summary", "comparative", "trends", "session"],
        help="Generate report and exit"
    )
    
    parser.add_argument(
        "--report-session",
        help="Session ID for session-specific report"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)"
    )
    
    # System options
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory containing sessions (default: output)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser

def initialize_system(output_dir: str, verbose: bool = False):
    """Initialize the RAG system components."""
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available. Cannot initialize system.")
        return None, None, None
    
    if verbose:
        print(f"🚀 Initializing Audio RAG System...")
        print(f"📁 Output directory: {output_dir}")
    
    # Initialize session manager
    session_manager = SessionManager(output_dir)
    
    if not session_manager.available_sessions:
        print(f"❌ No sessions found in {output_dir}/sessions/")
        print("   Make sure you have run the audio analysis pipeline first.")
        return None, None, None
    
    # Initialize RAG engine
    rag_engine = RAGEngine(output_dir)
    
    # Initialize report generator
    report_generator = ReportGenerator(output_dir)
    
    if verbose:
        print(f"✅ System initialized with {len(session_manager.available_sessions)} sessions")
    
    return session_manager, rag_engine, report_generator

def handle_session_listing(session_manager):
    """Handle --sessions command."""
    if not session_manager.available_sessions:
        print("❌ No sessions found")
        return
    
    print(f"\n📁 Available Sessions ({len(session_manager.available_sessions)}):")
    print("-" * 80)
    
    for session_id in sorted(session_manager.available_sessions):
        metadata = session_manager.get_session_metadata(session_id)
        files = session_manager.get_session_files(session_id)
        
        print(f"  {session_id}:")
        print(f"    Duration: {metadata.get('duration_seconds', 0):.1f}s")
        print(f"    Files: {len(files)} ({', '.join(files)})")
        
        if metadata.get('themes'):
            print(f"    Themes: {', '.join(metadata['themes'][:3])}")
        print()

def handle_query_operations(args, session_manager, rag_engine):
    """Handle query operations."""
    # Load sessions based on arguments
    sessions_to_load = []
    
    if args.load_all:
        sessions_to_load = list(session_manager.available_sessions)
    elif args.session:
        sessions_to_load = args.session
    elif args.pattern:
        sessions_to_load = session_manager.get_sessions_by_pattern(args.pattern)
    else:
        # Load first 5 sessions as default
        sessions_to_load = list(session_manager.available_sessions)[:5]
        print(f"⚠️  No sessions specified, loading first 5 sessions")
    
    if args.verbose:
        print(f"🔄 Loading {len(sessions_to_load)} sessions: {', '.join(sessions_to_load)}")
    
    rag_engine.load_sessions(sessions_to_load)
    
    # Mark sessions as loaded
    for session_id in sessions_to_load:
        session_manager.mark_session_loaded(session_id)
    
    results = None
    
    # Perform query operations
    if args.query:
        print(f"🔍 Query: {args.query}")
        results = rag_engine.query(args.query, max_results=args.max_results)
        
    elif args.smart_query:
        print(f"🤖 Smart Query: {args.smart_query}")
        response = rag_engine.smart_query(args.smart_query, max_results=args.max_results)
        print(f"\n🎯 AI Response:")
        print("-" * 60)
        print(response)
        return
        
    elif args.search:
        print(f"🔍 Search: {args.search}")
        # Parse search criteria
        filters = {}
        text_query = []
        
        for part in args.search.split():
            if ":" in part:
                key, value = part.split(":", 1)
                filters[key] = value
            else:
                text_query.append(part)
        
        text = " ".join(text_query) if text_query else None
        results = rag_engine.search_content(text_query=text, **filters)
    
    # Display results
    if results:
        print(f"\n📋 Found {len(results)} results:")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Session: {result.get('session_id', 'Unknown')}")
            print(f"   Score: {result.get('similarity_score', 0):.3f}")
            print(f"   Type: {result.get('content_type', 'Unknown')}")
            
            if result.get('timestamp'):
                print(f"   Time: {result['timestamp']}")
            if result.get('speaker'):
                print(f"   Speaker: {result['speaker']}")
            
            content = result.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   Content: {content}")
        
        # Save results if output file specified
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")
    
    else:
        print("❌ No results found")

def handle_report_generation(args, session_manager, report_generator):
    """Handle report generation."""
    if args.report == "session":
        if not args.report_session:
            print("❌ Please specify --report-session for session-specific report")
            return
        
        print(f"📊 Generating session report for: {args.report_session}")
        report = report_generator.generate_session_report(args.report_session)
        
    elif args.report == "summary":
        sessions = list(session_manager.available_sessions)
        print(f"📊 Generating summary report for {len(sessions)} sessions")
        report = report_generator.generate_summary_report(sessions)
        
    elif args.report == "comparative":
        sessions = list(session_manager.available_sessions)
        if len(sessions) < 2:
            print("❌ Need at least 2 sessions for comparative report")
            return
        
        print(f"📊 Generating comparative report for {len(sessions)} sessions")
        report = report_generator.generate_comparative_report(sessions)
        
    elif args.report == "trends":
        sessions = list(session_manager.available_sessions)
        print(f"📊 Generating trends report for {len(sessions)} sessions")
        report = report_generator.generate_trends_report(sessions)
    
    # Display report
    print(f"\n📊 Report: {report.get('title', 'Untitled')}")
    print(f"📅 Generated: {report.get('timestamp', 'Unknown')}")
    print("-" * 60)
    
    if report.get('summary'):
        print(f"\n📋 Summary:")
        print(report['summary'])
    
    if report.get('key_findings'):
        print(f"\n🔍 Key Findings:")
        for finding in report['key_findings']:
            print(f"  • {finding}")
    
    if report.get('statistics'):
        print(f"\n📈 Statistics:")
        for key, value in report['statistics'].items():
            print(f"  {key}: {value}")
    
    # Save report if output file specified
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report saved to: {args.output}")

def main():
    """Main entry point."""
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot start: Required modules are missing.")
        print("Please ensure all Python files are in the same directory and dependencies are installed.")
        sys.exit(1)
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize system
    session_manager, rag_engine, report_generator = initialize_system(
        args.output_dir, 
        args.verbose
    )
    
    if not session_manager:
        sys.exit(1)
    
    try:
        # Handle different operation modes
        if args.sessions:
            handle_session_listing(session_manager)
            
        elif args.query or args.smart_query or args.search:
            handle_query_operations(args, session_manager, rag_engine)
            
        elif args.report:
            handle_report_generation(args, session_manager, report_generator)
            
        else:
            # Default: Start interactive CLI
            print("🚀 Starting interactive CLI...")
            cli = AudioRAGCLI()
            
            # Apply any initial session selection
            if args.load_all:
                cli.session_manager.select_sessions()
                cli.rag_engine.load_sessions(list(cli.session_manager.available_sessions))
            elif args.session:
                cli.session_manager.select_sessions(session_ids=args.session)
                cli.rag_engine.load_sessions(args.session)
            elif args.pattern:
                sessions = cli.session_manager.get_sessions_by_pattern(args.pattern)
                cli.session_manager.select_sessions(session_ids=sessions)
                cli.rag_engine.load_sessions(sessions)
            
            # Set configuration
            cli.max_results = args.max_results
            
            # Start interactive loop
            cli.cmdloop()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
