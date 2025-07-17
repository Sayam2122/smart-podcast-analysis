# 🎙️ Podcast Analysis Pipeline with RAG System

## Quick Start Guide

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama for local LLM (optional but recommended)
# Follow instructions in models/ollama_setup.md
```

### 2. Basic Usage

#### Option A: Interactive Demo
```bash
python demo.py
```

#### Option B: Process Audio File Directly
```bash
# Through CLI
python rag_cli.py --process audio.mp3

# Through demo script
python demo.py --audio audio.mp3
```

#### Option C: Web Interface
```bash
streamlit run streamlit_app.py
```

### 3. Query Examples

#### CLI Queries
```bash
# Interactive mode
python rag_cli.py

# Single query
python rag_cli.py --query "summarize the main points"
python rag_cli.py --query "what did the host say about AI"
python rag_cli.py --query "happy moments in the conversation"
```

#### Programmatic Usage
```python
from pipeline.pipeline_runner import process_podcast
from rag_system.query_processor import QueryProcessor
from rag_system.vector_database import VectorDatabase

# Process audio
results = process_podcast("audio.mp3")

# Add to RAG system
vector_db = VectorDatabase()
session_id = vector_db.add_podcast_session(results)

# Query content
query_processor = QueryProcessor(vector_db)
result = query_processor.process_query("what are the main topics?")
```

## System Architecture

### Pipeline Modules (`/pipeline/`)

1. **Audio Ingestion** (`audio_ingestion.py`)
   - Multi-format support (MP3, WAV, M4A, FLAC)
   - Normalization to 16kHz mono PCM
   - Noise gating and validation

2. **Transcription** (`transcription.py`)
   - Whisper/Faster-Whisper integration
   - Confidence scoring and word-level timestamps
   - SRT subtitle export

3. **Speaker Diarization** (`diarization.py`)
   - Pyannote-Audio speaker separation
   - Speaker clustering and identification
   - Confidence scores and fallback methods

4. **Emotion Detection** (`emotion_detection.py`)
   - Dual-mode: text + audio emotion analysis
   - DistilRoBERTa for text emotions
   - Wav2Vec2 for audio emotions

5. **Semantic Segmentation** (`semantic_segmentation.py`)
   - SentenceTransformers embeddings
   - Topic modeling (LDA, TF-IDF)
   - Clustering-based segmentation

6. **Summarization** (`summarization.py`)
   - Local Ollama LLM integration
   - Block-wise and overall summaries
   - Fallback extractive summarization

7. **Pipeline Runner** (`pipeline_runner.py`)
   - Session-based processing
   - Resume capability
   - Progress tracking and caching

### RAG System (`/rag_system/`)

1. **Vector Database** (`vector_database.py`)
   - ChromaDB for persistent storage
   - Multi-level indexing (blocks, segments, summaries)
   - Metadata filtering and search

2. **Query Processor** (`query_processor.py`)
   - Natural language query parsing
   - Intent detection and parameter extraction
   - Context-aware ranking and filtering

### Interfaces

1. **CLI Interface** (`rag_cli.py`)
   - Interactive query sessions
   - Session management
   - Audio processing integration

2. **Web Interface** (`streamlit_app.py`)
   - Visual query interface
   - Analytics and visualizations
   - Session management UI
   - Audio upload and processing

### Utilities (`/utils/`)

- **Logger** (`logger.py`): Session-based logging with loguru
- **File Utils** (`file_utils.py`): I/O operations and caching

## Key Features

### 🎯 **Comprehensive Analysis**
- **Audio Processing**: Multi-format support with intelligent normalization
- **Speech Recognition**: High-accuracy transcription with timestamps
- **Speaker Analysis**: Automatic speaker identification and separation
- **Emotion Detection**: Dual-mode emotion analysis (text + audio)
- **Content Segmentation**: Semantic topic-based segmentation
- **AI Summarization**: Local LLM-powered summaries and insights

### 🔍 **Advanced Search & RAG**
- **Vector Search**: Semantic similarity using SentenceTransformers
- **Natural Language Queries**: Intelligent query parsing and intent detection
- **Multi-Modal Filtering**: Filter by speaker, emotion, time, content type
- **Context Retrieval**: Conversation context around search results
- **Session Management**: Organize and explore multiple podcast sessions

### 🚀 **Production-Ready Features**
- **Session-Based Processing**: Resume interrupted processing
- **Memory Efficiency**: Lazy loading and streaming processing
- **Error Handling**: Comprehensive fallback mechanisms
- **Local Processing**: No API keys or cloud dependencies required
- **Scalable Storage**: Persistent vector database with ChromaDB

### 🎨 **Multiple Interfaces**
- **CLI**: Interactive command-line interface for power users
- **Web UI**: Beautiful Streamlit interface with visualizations
- **Programmatic**: Python API for integration into other systems
- **Demo Mode**: Guided demonstration of all features

## Example Use Cases

### 📚 **Content Creation**
```bash
# Get key quotes for social media
python rag_cli.py --query "most interesting quotes from the guest"

# Extract main topics for blog posts
python rag_cli.py --query "what topics were covered in detail"
```

### 🎙️ **Podcast Analysis**
```bash
# Analyze speaking patterns
python rag_cli.py --query "who spoke the most in this episode"

# Find emotional peaks
python rag_cli.py --query "most exciting moments in the conversation"
```

### 📖 **Research & Learning**
```bash
# Topic deep-dive
python rag_cli.py --query "everything discussed about machine learning"

# Speaker insights
python rag_cli.py --query "what did the expert say about the future"
```

### 🔗 **Content Discovery**
```bash
# Cross-episode analysis
python rag_cli.py --query "similar topics across all sessions"

# Trend analysis
python rag_cli.py --query "how has the discussion evolved over time"
```

## Technical Specifications

### **Models Used**
- **Transcription**: OpenAI Whisper / Faster-Whisper (base, small, medium, large)
- **Diarization**: Pyannote-Audio speaker separation models
- **Text Emotions**: DistilRoBERTa emotion classification
- **Audio Emotions**: Wav2Vec2 emotion recognition
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
- **Summarization**: Ollama local LLMs (Mistral 7B, Llama3.2 3B)

### **Performance**
- **Processing Speed**: Typically 2-5x real-time (depends on model size)
- **Memory Usage**: Optimized with lazy loading and streaming
- **Storage**: Efficient vector storage with ChromaDB
- **Scalability**: Session-based architecture supports large collections

### **Output Formats**
- **JSON**: Complete structured results
- **SRT**: Subtitle files for video editing
- **TXT**: Human-readable summaries
- **CSV**: Metadata exports for analysis

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install chromadb sentence-transformers
   ```

2. **Ollama Connection Failed**
   ```bash
   # Install and start Ollama
   # See models/ollama_setup.md for instructions
   ```

3. **Audio Processing Errors**
   ```bash
   # Install ffmpeg for audio format support
   # Windows: Download from ffmpeg.org
   # Mac: brew install ffmpeg
   # Linux: apt install ffmpeg
   ```

4. **Memory Issues**
   ```bash
   # Use smaller models
   # Whisper: 'base' instead of 'large'
   # Embeddings: 'all-MiniLM-L6-v2' instead of 'all-mpnet-base-v2'
   ```

### **Getting Help**

1. **Check Logs**: Session logs are stored in `output/sessions/session_*/`
2. **Demo Mode**: Run `python demo.py` for guided troubleshooting
3. **Verbose Output**: Set environment variable `LOG_LEVEL=DEBUG`

## Development & Customization

### Adding New Query Types
```python
# In rag_system/query_processor.py
def _initialize_query_patterns(self):
    patterns = {
        'custom_queries': [
            r'find (.+?) discussions',
            r'locate (.+?) mentions'
        ]
    }
    return patterns
```

### Custom Emotion Models
```python
# In pipeline/emotion_detection.py
emotion_detector = EmotionDetection(
    text_model='your-custom-model',
    audio_model='your-audio-model'
)
```

### New Output Formats
```python
# In pipeline/pipeline_runner.py
def _save_custom_format(self, results, path):
    # Implement your custom export format
    pass
```

---

**🎉 Ready to analyze your podcasts! Start with `python demo.py` for a guided tour of all features.**
