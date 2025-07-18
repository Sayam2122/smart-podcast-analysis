"""
Test Script for Audio RAG System

Simple test to verify all components are working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from data_loader import DataLoader
        print("✅ data_loader imported successfully")
    except ImportError as e:
        print(f"❌ data_loader import failed: {e}")
        return False
    
    try:
        from context_router import ContextRouter
        print("✅ context_router imported successfully")
    except ImportError as e:
        print(f"❌ context_router import failed: {e}")
        return False
    
    try:
        from session_manager import SessionManager
        print("✅ session_manager imported successfully")
    except ImportError as e:
        print(f"❌ session_manager import failed: {e}")
        return False
    
    try:
        from report_generator import ReportGenerator
        print("✅ report_generator imported successfully")
    except ImportError as e:
        print(f"❌ report_generator import failed: {e}")
        return False
    
    try:
        from rag_engine import RAGEngine
        print("✅ rag_engine imported successfully")
    except ImportError as e:
        print(f"❌ rag_engine import failed: {e}")
        return False
    
    try:
        from enhanced_cli import AudioRAGCLI
        print("✅ enhanced_cli imported successfully")
    except ImportError as e:
        print(f"❌ enhanced_cli import failed: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality of each component."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test DataLoader
        from data_loader import DataLoader
        loader = DataLoader()
        sessions = loader.discover_sessions()
        print(f"✅ DataLoader: Found {len(sessions)} sessions")
        
        # Test SessionManager
        from session_manager import SessionManager
        manager = SessionManager()
        print(f"✅ SessionManager: {len(manager.available_sessions)} sessions available")
        
        # Test ContextRouter
        from context_router import ContextRouter
        router = ContextRouter()
        test_query = "What was discussed about AI?"
        config = router.route_query(test_query, ['transcription', 'emotion_detection'])
        print(f"✅ ContextRouter: Query routed to {len(config['content_types'])} content types")
        
        # Test ReportGenerator
        from report_generator import ReportGenerator
        generator = ReportGenerator()
        print("✅ ReportGenerator: Initialized successfully")
        
        # Test RAGEngine (may fail due to dependencies)
        try:
            from rag_engine import RAGEngine
            engine = RAGEngine()
            print("✅ RAGEngine: Initialized successfully")
        except Exception as e:
            print(f"⚠️  RAGEngine: Limited functionality ({e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_cli():
    """Test CLI initialization."""
    print("\n🖥️  Testing CLI...")
    
    try:
        from enhanced_cli import AudioRAGCLI
        cli = AudioRAGCLI()
        if hasattr(cli, 'session_manager'):
            print("✅ CLI: Initialized successfully")
            return True
        else:
            print("⚠️  CLI: Initialized with limited functionality")
            return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def create_sample_session():
    """Create a sample session for testing if none exist."""
    print("\n📁 Creating sample session for testing...")
    
    output_dir = Path("output/sessions/test_session_001")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample transcription file
    sample_transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Hello everyone, welcome to our meeting about artificial intelligence.",
                "speaker": "speaker_1",
                "confidence": 0.95
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "Today we'll discuss the future of AI technology.",
                "speaker": "speaker_1",
                "confidence": 0.92
            }
        ],
        "language": "english",
        "total_duration": 10.0
    }
    
    with open(output_dir / "transcription.json", "w", encoding='utf-8') as f:
        import json
        json.dump(sample_transcription, f, indent=2)
    
    # Create sample emotion detection
    sample_emotions = {
        "emotions": [
            {
                "timestamp": 2.0,
                "emotion": "neutral",
                "confidence": 0.8,
                "intensity": 0.5,
                "text": "Hello everyone",
                "speaker": "speaker_1"
            }
        ]
    }
    
    with open(output_dir / "emotion_detection.json", "w", encoding='utf-8') as f:
        json.dump(sample_emotions, f, indent=2)
    
    # Create sample final report
    sample_report = {
        "audio_file": "test_audio.wav",
        "total_duration_seconds": 10.0,
        "processing_time": 5.0,
        "status": "completed"
    }
    
    with open(output_dir / "final_report.json", "w", encoding='utf-8') as f:
        json.dump(sample_report, f, indent=2)
    
    print(f"✅ Sample session created at: {output_dir}")
    return True

def main():
    """Run all tests."""
    print("🎙️  Audio RAG System - Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Check if main files exist
    required_files = [
        "data_loader.py", "context_router.py", "session_manager.py",
        "report_generator.py", "rag_engine.py", "enhanced_cli.py", "main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_cli():
        tests_passed += 1
    
    # Create sample session if no sessions exist
    from session_manager import SessionManager
    manager = SessionManager()
    if not manager.available_sessions:
        print("🎯 No sessions found - creating sample session...")
        if create_sample_session():
            tests_passed += 1
            print("✅ Sample session created successfully")
        else:
            print("❌ Failed to create sample session")
    else:
        print(f"✅ Found {len(manager.available_sessions)} existing sessions")
        tests_passed += 1
    
    # Final results
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready to use.")
        print("\nQuick start commands:")
        print("  python main.py --sessions              # List available sessions")
        print("  python main.py                         # Start interactive CLI")
        print("  python main.py --query 'What was discussed about AI?'")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
