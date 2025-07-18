"""
Enhanced CLI Interface for Audio RAG System

Provides comprehensive command-line interface for querying audio analysis data,
managing sessions, generating reports, and interactive exploration.
"""

import cmd
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Import our RAG system components
try:
    from rag_engine import RAGEngine
    from session_manager import SessionManager
    from report_generator import ReportGenerator
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    RAGEngine = None
    SessionManager = None
    ReportGenerator = None
    IMPORTS_AVAILABLE = False

class AudioRAGCLI(cmd.Cmd):
    """
    Interactive command-line interface for the Audio RAG system.
    """
    
    intro = """
🎙️  Audio RAG System - Interactive CLI
=====================================

Welcome to the Audio RAG system! This interface allows you to:
• Query transcriptions, emotions, and semantic segments
• Manage and filter sessions
• Generate comprehensive reports
• Explore your audio analysis data

Type 'help' or '?' for available commands.
Type 'status' to see system status.
Type 'sessions' to see available sessions.
"""
    
    prompt = "AudioRAG> "
    
    def __init__(self):
        super().__init__()
        
        if not IMPORTS_AVAILABLE:
            print("❌ Required modules not available. Please ensure all dependencies are installed.")
            self.session_manager = None
            self.rag_engine = None
            self.report_generator = None
            return
        
        self.session_manager = SessionManager()
        self.rag_engine = None
        self.report_generator = ReportGenerator()
        
        # Configuration
        self.auto_load = False
        self.max_results = 10
        self.context_size = 5
        
        # Initialize RAG engine if sessions are available
        if self.session_manager.available_sessions:
            print(f"🚀 Initializing RAG engine with {len(self.session_manager.available_sessions)} sessions...")
            self.rag_engine = RAGEngine()
            self.rag_engine.load_sessions(list(self.session_manager.available_sessions)[:5])  # Load first 5 sessions
    
    # === Session Management Commands ===
    
    def do_sessions(self, arg):
        """List available sessions with metadata."""
        if not self.session_manager.available_sessions:
            print("❌ No sessions found in output/sessions/ directory")
            return
        
        print(f"\n📁 Available Sessions ({len(self.session_manager.available_sessions)}):")
        print("-" * 80)
        
        for session_id in sorted(self.session_manager.available_sessions):
            metadata = self.session_manager.get_session_metadata(session_id)
            files = self.session_manager.get_session_files(session_id)
            
            status_indicators = []
            if session_id in self.session_manager.loaded_sessions:
                status_indicators.append("🟢 LOADED")
            if session_id in self.session_manager.selected_sessions:
                status_indicators.append("🎯 SELECTED")
            
            status = " ".join(status_indicators) if status_indicators else "⚪ Available"
            
            print(f"  {session_id}: {status}")
            print(f"    Duration: {metadata.get('duration_seconds', 0):.1f}s")
            print(f"    Files: {', '.join(files)}")
            
            if metadata.get('themes'):
                print(f"    Themes: {', '.join(metadata['themes'][:3])}")
            print()
    
    def do_select(self, arg):
        """
        Select sessions for querying.
        Usage: 
          select all                    - Select all sessions
          select session1 session2      - Select specific sessions
          select pattern:keyword        - Select sessions containing keyword
          select duration:min_seconds   - Select sessions longer than X seconds
          select files:transcription    - Select sessions with specific files
        """
        if not arg.strip():
            print("❌ Please specify sessions to select. Use 'help select' for usage.")
            return
        
        parts = arg.strip().split()
        
        if parts[0] == "all":
            selected = self.session_manager.select_sessions()
            
        elif parts[0].startswith("pattern:"):
            pattern = parts[0].split(":", 1)[1]
            selected = self.session_manager.select_sessions(pattern=pattern)
            
        elif parts[0].startswith("duration:"):
            min_duration = int(parts[0].split(":", 1)[1])
            selected = self.session_manager.select_sessions(min_duration=min_duration)
            
        elif parts[0].startswith("files:"):
            required_file = parts[0].split(":", 1)[1] + ".json"
            selected = self.session_manager.select_sessions(has_file=required_file)
            
        else:
            # Treat as explicit session IDs
            selected = self.session_manager.select_sessions(session_ids=parts)
        
        if selected:
            print(f"✅ Selected {len(selected)} sessions")
            if self.rag_engine:
                print("🔄 Reloading RAG engine with selected sessions...")
                self.rag_engine.load_sessions(selected)
        else:
            print("❌ No sessions matched the selection criteria")
    
    def do_load(self, arg):
        """
        Load sessions into the RAG engine.
        Usage: load [session_ids...] or load selected
        """
        if not self.rag_engine:
            print("❌ RAG engine not initialized")
            return
        
        if not arg.strip() or arg.strip() == "selected":
            # Load selected sessions
            sessions_to_load = list(self.session_manager.selected_sessions)
            if not sessions_to_load:
                sessions_to_load = list(self.session_manager.available_sessions)[:5]
                print("⚠️  No sessions selected, loading first 5 available sessions")
        else:
            # Load specific sessions
            sessions_to_load = arg.strip().split()
        
        print(f"🔄 Loading {len(sessions_to_load)} sessions into RAG engine...")
        self.rag_engine.load_sessions(sessions_to_load)
        
        for session_id in sessions_to_load:
            self.session_manager.mark_session_loaded(session_id)
        
        print(f"✅ Loaded sessions: {', '.join(sessions_to_load)}")
    
    # === Query Commands ===
    
    def do_query(self, arg):
        """
        Perform intelligent query across loaded sessions.
        Usage: query What topics were discussed about AI?
        """
        if not self.rag_engine:
            print("❌ RAG engine not initialized")
            return
        
        if not arg.strip():
            print("❌ Please provide a query. Example: query What topics were discussed?")
            return
        
        print(f"🔍 Searching: {arg}")
        try:
            results = self.rag_engine.query(arg, max_results=self.max_results)
            
            if results:
                print(f"\n📋 Found {len(results)} relevant results:")
                print("-" * 60)
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Session: {result.get('session_id', 'Unknown')}")
                    print(f"   Score: {result.get('similarity_score', 0):.3f}")
                    print(f"   Type: {result.get('content_type', 'Unknown')}")
                    print(f"   Content: {result.get('content', '')[:200]}...")
                    
                    if result.get('timestamp'):
                        print(f"   Time: {result['timestamp']}")
                    if result.get('speaker'):
                        print(f"   Speaker: {result['speaker']}")
            else:
                print("❌ No relevant results found")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    def do_smart_query(self, arg):
        """
        Perform LLM-enhanced query with domain and emotion awareness.
        Usage: smart_query Summarize the main discussion points
        """
        if not self.rag_engine:
            print("❌ RAG engine not initialized")
            return
        
        if not arg.strip():
            print("❌ Please provide a query for LLM analysis")
            return
        
        print(f"🤖 Enhanced LLM Query: {arg}")
        
        try:
            # Use enhanced smart query
            result = self.rag_engine.enhanced_smart_query(
                query=arg,
                session_id=None,  # Search all sessions
                max_results=self.max_results
            )
            
            # Display enhanced results
            print(f"\n🎯 Domain: {result['domain']}")
            
            emotion_context = result['emotion_context']
            if emotion_context['primary_emotion'] != 'neutral':
                print(f"💭 Emotional Context: {emotion_context['primary_emotion']} (intensity: {emotion_context['intensity']})")
            
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"📝 Sources: {result['source_count']}")
            
            print(f"\n🎯 AI Response:")
            print("-" * 60)
            print(result['answer'])
            
            # Show routing info if verbose mode is available
            if hasattr(self, 'verbose') and self.verbose:
                routing = result['routing_info']
                print(f"\n🧭 Routing Details:")
                print(f"  Focuses: {routing.get('focuses', [])}")
                print(f"  Strategy: {routing.get('strategy', {}).get('search_type', 'unknown')}")
                if routing.get('filters'):
                    print(f"  Filters: {routing['filters']}")
            
        except Exception as e:
            print(f"❌ Enhanced smart query failed: {e}")
            # Fallback to original method
            try:
                response = self.rag_engine.smart_query(arg, max_results=self.max_results)
                print(f"\n🎯 AI Response (fallback):")
                print("-" * 60)
                print(response)
            except Exception as e2:
                print(f"❌ Fallback query also failed: {e2}")
    
    def do_search(self, arg):
        """
        Search for specific content across sessions.
        Usage: search content_type:transcription speaker:John text:meeting
        """
        if not self.rag_engine:
            print("❌ RAG engine not initialized")
            return
        
        if not arg.strip():
            print("❌ Please provide search criteria")
            return
        
        # Parse search criteria
        filters = {}
        text_query = []
        
        for part in arg.split():
            if ":" in part:
                key, value = part.split(":", 1)
                filters[key] = value
            else:
                text_query.append(part)
        
        text = " ".join(text_query) if text_query else ""
        
        print(f"🔍 Searching with filters: {filters}")
        if text:
            print(f"🔍 Text query: {text}")
        
        try:
            results = self.rag_engine.search_content(
                text_query=text if text else None,
                **filters
            )
            
            if results:
                print(f"\n📋 Found {len(results)} matching results:")
                self._display_search_results(results)
            else:
                print("❌ No results found matching criteria")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    # === Report Generation Commands ===
    
    def do_report(self, arg):
        """
        Generate comprehensive reports.
        Usage: 
          report session [session_id]     - Session-specific report
          report comparative             - Compare multiple sessions
          report summary                 - Executive summary
          report trends                  - Trend analysis
        """
        parts = arg.strip().split()
        if not parts:
            print("❌ Please specify report type. Use 'help report' for options.")
            return
        
        report_type = parts[0]
        
        try:
            if report_type == "session":
                session_id = parts[1] if len(parts) > 1 else None
                if not session_id:
                    if len(self.session_manager.selected_sessions) == 1:
                        session_id = list(self.session_manager.selected_sessions)[0]
                    else:
                        print("❌ Please specify session ID or select exactly one session")
                        return
                
                print(f"📊 Generating session report for: {session_id}")
                report = self.report_generator.generate_session_report(session_id)
                self._display_report(report)
                
            elif report_type == "comparative":
                if len(self.session_manager.selected_sessions) < 2:
                    print("❌ Please select at least 2 sessions for comparison")
                    return
                
                print(f"📊 Generating comparative report for {len(self.session_manager.selected_sessions)} sessions")
                report = self.report_generator.generate_comparative_report(
                    list(self.session_manager.selected_sessions)
                )
                self._display_report(report)
                
            elif report_type == "summary":
                sessions = list(self.session_manager.selected_sessions) or list(self.session_manager.available_sessions)
                print(f"📊 Generating summary report for {len(sessions)} sessions")
                report = self.report_generator.generate_summary_report(sessions)
                self._display_report(report)
                
            elif report_type == "trends":
                sessions = list(self.session_manager.selected_sessions) or list(self.session_manager.available_sessions)
                print(f"📊 Generating trends report for {len(sessions)} sessions")
                report = self.report_generator.generate_trends_report(sessions)
                self._display_report(report)
                
            else:
                print(f"❌ Unknown report type: {report_type}")
                
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
    
    # === Utility Commands ===
    
    def do_status(self, arg):
        """Show system status and configuration."""
        summary = self.session_manager.get_sessions_summary()
        
        print("\n🔧 System Status:")
        print("-" * 40)
        print(f"RAG Engine: {'✅ Ready' if self.rag_engine else '❌ Not initialized'}")
        print(f"Sessions Available: {summary['total_sessions']}")
        print(f"Sessions Selected: {summary['selected_sessions']}")
        print(f"Sessions Loaded: {summary['loaded_sessions']}")
        print(f"Total Duration: {summary['total_duration_hours']} hours")
        print(f"Max Results: {self.max_results}")
        print(f"Context Size: {self.context_size}")
        
        if summary['file_availability']:
            print(f"\n📁 File Availability:")
            for file_type, count in summary['file_availability'].items():
                print(f"  {file_type}: {count} sessions")
    
    def do_config(self, arg):
        """
        Configure system settings.
        Usage: 
          config max_results 20     - Set max results per query
          config context_size 10    - Set context window size
        """
        parts = arg.strip().split()
        if len(parts) != 2:
            print("❌ Usage: config <setting> <value>")
            return
        
        setting, value = parts
        
        if setting == "max_results":
            self.max_results = int(value)
            print(f"✅ Max results set to {self.max_results}")
        elif setting == "context_size":
            self.context_size = int(value)
            print(f"✅ Context size set to {self.context_size}")
        else:
            print(f"❌ Unknown setting: {setting}")
    
    def do_export(self, arg):
        """
        Export query results or reports.
        Usage: export results results.json
        """
        print("📤 Export functionality - to be implemented")
    
    def do_clear(self, arg):
        """Clear the screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def do_exit(self, arg):
        """Exit the CLI."""
        print("👋 Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the CLI."""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl+D."""
        print("\n👋 Goodbye!")
        return True
    
    # === Helper Methods ===
    
    def _display_search_results(self, results: List[Dict]):
        """Display search results in a formatted way."""
        for i, result in enumerate(results[:self.max_results], 1):
            print(f"\n{i}. Session: {result.get('session_id', 'Unknown')}")
            print(f"   Type: {result.get('content_type', 'Unknown')}")
            
            if result.get('timestamp'):
                print(f"   Time: {result['timestamp']}")
            if result.get('speaker'):
                print(f"   Speaker: {result['speaker']}")
            if result.get('emotion'):
                print(f"   Emotion: {result['emotion']}")
            
            content = result.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   Content: {content}")
    
    def _display_report(self, report: Dict):
        """Display a generated report."""
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
    
    def emptyline(self):
        """Override empty line behavior to do nothing."""
        pass

    def do_enhanced_query(self, arg):
        """
        Perform advanced query with full domain and emotion analysis display.
        Usage: enhanced_query What emotional patterns emerged during the technology discussion?
        """
        if not self.rag_engine:
            print("❌ RAG engine not initialized")
            return
        
        if not arg.strip():
            print("❌ Please provide a query for enhanced analysis")
            return
        
        print(f"🚀 Enhanced Query Analysis: {arg}")
        print("=" * 80)
        
        try:
            # Use enhanced smart query
            result = self.rag_engine.enhanced_smart_query(
                query=arg,
                session_id=None,
                max_results=self.max_results
            )
            
            # Display comprehensive analysis
            print(f"\n📊 Analysis Overview:")
            print(f"  🎯 Domain: {result['domain']}")
            print(f"  📊 Confidence: {result['confidence']:.2f}")
            print(f"  📝 Source Count: {result['source_count']}")
            
            # Emotion analysis
            emotion_context = result['emotion_context']
            print(f"\n💭 Emotional Analysis:")
            print(f"  Primary Emotion: {emotion_context['primary_emotion']}")
            print(f"  Intensity: {emotion_context['intensity']}")
            print(f"  Confidence: {emotion_context['confidence']:.2f}")
            if emotion_context.get('seeking_emotions'):
                print(f"  Seeking Emotions: {', '.join(emotion_context['seeking_emotions'])}")
            
            # Routing details
            routing = result['routing_info']
            print(f"\n🧭 Query Routing:")
            print(f"  Focuses: {', '.join(routing.get('focuses', []))}")
            print(f"  Strategy: {routing.get('strategy', {}).get('search_type', 'unknown')}")
            print(f"  Content Types: {', '.join(routing.get('content_types', [])[:3])}")
            print(f"  Complexity: {routing.get('query_complexity', 'unknown')}")
            
            if routing.get('filters'):
                print(f"  Active Filters: {list(routing['filters'].keys())}")
            
            # Main answer
            print(f"\n🎯 Comprehensive Answer:")
            print("-" * 60)
            print(result['answer'])
            
            # Show top sources with details
            if result['sources']:
                print(f"\n📚 Top Evidence Sources:")
                print("-" * 40)
                for i, source in enumerate(result['sources'][:3], 1):
                    metadata = source.get('metadata', {})
                    print(f"\n{i}. Similarity: {source.get('similarity_score', 0):.3f}")
                    print(f"   Type: {source.get('content_type', 'unknown')}")
                    
                    if 'speaker' in metadata:
                        print(f"   Speaker: {metadata['speaker']}")
                    if 'start_time' in metadata:
                        print(f"   Time: {metadata['start_time']:.1f}s")
                    if 'combined_emotion' in metadata:
                        emotion = metadata['combined_emotion']
                        print(f"   Emotion: {emotion.get('emotion', 'neutral')} ({emotion.get('confidence', 0):.2f})")
                    if 'themes' in metadata and metadata['themes']:
                        print(f"   Themes: {', '.join(metadata['themes'][:2])}")
                    
                    # Truncated content
                    content = source['content']
                    if len(content) > 150:
                        content = content[:150] + "..."
                    print(f"   Content: {content}")
            
            print(f"\n{'='*80}")
            
        except Exception as e:
            print(f"❌ Enhanced query analysis failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point for the CLI."""
    try:
        cli = AudioRAGCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")

if __name__ == "__main__":
    main()
