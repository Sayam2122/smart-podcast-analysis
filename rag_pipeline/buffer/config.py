# /podcast_rag_project/config.py

import os

# --- Model Configuration ---
# Specifies the models to be used for embedding generation and language processing.
# Ensure the OLLAMA_MODEL is downloaded and available in your local Ollama instance.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
OLLAMA_MODEL = 'mistral:latest'

# --- Directory Paths ---
# Defines the folder structure for the project. Using absolute paths ensures
# the script can be run from any location without breaking file references.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EPISODES_DIR = os.path.join(DATA_DIR, 'episodes') # Main directory containing all episode folders
VECTOR_STORE_DIR = os.path.join(BASE_DIR, 'vector_store')
CONVERSATION_HISTORY_DIR = os.path.join(BASE_DIR, 'conversation_history')
CONTENT_ASSETS_DIR = os.path.join(BASE_DIR, 'content_assets')

# --- File Names ---
# A dictionary defining the exact filenames the system expects to find inside
# each episode's dedicated folder. This ensures consistency and simplifies the
# data loading process.
REQUIRED_FILES = {
    "emotion": "emotion_detection.json",
    "semantic": "semantic_segmentation.json",
    "final_report": "final_report.json",
    "summarization": "summarization.json", # Using for detailed block-level insights
}

# --- Enhanced Chunking Configuration ---
ENABLE_ENHANCED_CHUNKING = True
ENABLE_SEMANTIC_ANALYSIS = True
ENABLE_CONTENT_EXTRACTION = True

# --- Response Configuration ---
RESPONSE_CONFIDENCE_THRESHOLD = 0.7
MAX_CONVERSATION_HISTORY = 10
ENABLE_FOLLOWUP_SUGGESTIONS = True