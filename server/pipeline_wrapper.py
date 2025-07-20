#!/usr/bin/env python3
"""
Pipeline wrapper script for Node.js server integration.
This script runs the main pipeline and provides progress updates.
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
sys.path.insert(0, str(parent_dir / 'pipeline'))

def main():
    parser = argparse.ArgumentParser(description='Pipeline Wrapper')
    parser.add_argument('--audio-file', required=True, help='Path to audio file')
    parser.add_argument('--session-id', required=True, help='Session ID')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--config', help='JSON configuration string')
    parser.add_argument('--resume', action='store_true', help='Resume existing session')
    
    args = parser.parse_args()
    
    try:
        from pipeline.pipeline_runner import PipelineRunner
        
        print(f"STEP: Initializing pipeline")
        print(f"PROGRESS: 0")
        
        # Parse config if provided
        config = {}
        if args.config:
            config = json.loads(args.config)
        
        # Initialize pipeline runner
        runner = PipelineRunner(
            output_dir=args.output_dir,
            session_id=args.session_id,
            config=config
        )
        
        print(f"STEP: Starting audio processing")
        print(f"PROGRESS: 5")
        
        # Process the audio file
        results = runner.process_audio_file(
            audio_file_path=args.audio_file,
            resume=args.resume
        )
        
        print(f"STEP: Pipeline completed")
        print(f"PROGRESS: 100")
        
        # Output results summary
        summary = {
            'session_id': args.session_id,
            'status': 'completed',
            'audio_file': args.audio_file,
            'total_segments': len(results.get('enriched_segments', [])),
            'total_blocks': len(results.get('semantic_blocks', [])),
            'processing_time': results.get('final_report', {}).get('session_info', {}).get('total_processing_time', 0)
        }
        
        print(f"RESULT: {json.dumps(summary)}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
