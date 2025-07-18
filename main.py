"""
Smart Audio RAG System - Main Entry Point

This is the main entry point for the Smart Audio RAG System.
Run this file to start the interactive CLI interface.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from rag_system.cli import cli
    
    if __name__ == "__main__":
        # Set up environment
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Avoid tokenizer warnings
        
        # Run the CLI
        cli()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔧 Make sure you have installed all dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error starting Smart Audio RAG System: {e}")
    sys.exit(1)
