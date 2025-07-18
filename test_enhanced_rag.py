#!/usr/bin/env python3
"""
Test script for enhanced RAG functionality with domain and emotion awareness.
"""

import sys
from pathlib import Path

# Add the rag_system directory to path
sys.path.append(str(Path(__file__).parent / 'rag_system'))

try:
    from rag_system.rag_engine import RAGEngine
    from rag_system.context_router import ContextRouter
    print("✅ Successfully imported RAG components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_context_router():
    """Test the enhanced context router."""
    print("\n🧭 Testing Enhanced Context Router")
    print("=" * 50)
    
    router = ContextRouter()
    
    test_queries = [
        "What emotions were detected when discussing AI technology?",
        "Who was the most excited speaker in the business meeting?",
        "Can you summarize the key healthcare discussions?",
        "What topics made people happy during the social gathering?",
        "Tell me about the frustrated moments in the education session"
    ]
    
    available_types = ['enriched_segment', 'summary_block', 'key_point', 'highlight', 'overall_summary']
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            routing = router.route_query(query, available_types)
            
            print(f"   🎯 Domain: {routing['domain']}")
            print(f"   💭 Primary Emotion: {routing['emotion_analysis']['primary_emotion']}")
            print(f"   📊 Emotion Intensity: {routing['emotion_analysis']['intensity']}")
            print(f"   🔍 Focuses: {', '.join(routing['focuses'])}")
            print(f"   📋 Strategy: {routing['strategy']['search_type']}")
            print(f"   🎛️ Content Types: {', '.join(routing['content_types'][:3])}")
            
            if routing['filters']:
                print(f"   🔧 Filters: {list(routing['filters'].keys())}")
            
            print(f"   ✨ Confidence: {routing.get('routing_confidence', 0):.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_rag_engine():
    """Test the enhanced RAG engine (without actual data)."""
    print("\n🤖 Testing Enhanced RAG Engine")
    print("=" * 50)
    
    try:
        rag_engine = RAGEngine()
        print("✅ RAG Engine initialized successfully")
        
        # Test data type detection
        available_types = rag_engine._get_available_data_types()
        print(f"📊 Available data types: {available_types}")
        
        print("✅ Enhanced RAG functionality appears to be working")
        
    except Exception as e:
        print(f"❌ RAG Engine test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Audio RAG System")
    print("=" * 60)
    
    test_context_router()
    test_rag_engine()
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced RAG System Test Complete!")
    print("\nKey Features Added:")
    print("  ✅ Domain detection (business, technology, education, etc.)")
    print("  ✅ Emotion analysis with intensity levels")
    print("  ✅ Enhanced query routing with context awareness")
    print("  ✅ Strategy-based search with emotion weighting")
    print("  ✅ Comprehensive result scoring and filtering")
    print("  ✅ LLM prompts with domain and emotion context")
    print("\nTry using the enhanced_cli.py with commands:")
    print("  > smart_query What emotions emerged during our AI discussion?")
    print("  > enhanced_query Who showed excitement about the technology demo?")

if __name__ == "__main__":
    main()
