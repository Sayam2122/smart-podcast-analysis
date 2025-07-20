#!/usr/bin/env python3
"""
RAG integration script for processed audio sessions.
This script integrates pipeline results with the RAG system.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'rag_system'))

def main():
    parser = argparse.ArgumentParser(description='RAG Integration')
    parser.add_argument('--session-id', required=True, help='Session ID')
    parser.add_argument('--sessions-dir', required=True, help='Sessions directory')
    
    args = parser.parse_args()
    
    try:
        from rag_system.core import PodcastRAGCore
        from rag_system.indexer import ContentIndexer
        
        print(f"STEP: Initializing RAG system")
        
        # Initialize RAG core
        rag_core = PodcastRAGCore()
        indexer = ContentIndexer(rag_core)
        
        # Find session directory
        session_dir = Path(args.sessions_dir) / f"session_{args.session_id}"
        
        if not session_dir.exists():
            raise Exception(f"Session directory not found: {session_dir}")
        
        print(f"STEP: Indexing session content")
        
        # Index the session
        results = indexer.index_session_directory(str(session_dir))
        
        print(f"STEP: RAG integration completed")
        
        # Output integration summary
        summary = {
            'session_id': args.session_id,
            'indexed_segments': results.get('total_segments', 0),
            'indexed_blocks': results.get('total_blocks', 0),
            'rag_ready': True
        }
        
        print(f"RESULT: {json.dumps(summary)}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
