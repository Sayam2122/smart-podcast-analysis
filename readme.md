# Podcast Audio Analysis Pipeline with RAG System

A complete, modular, memory-efficient, and real-time podcast/audio analysis pipeline with advanced speaker diarization, emotion detection, transcription, and topic modeling. Integrated with a fully searchable and interactive RAG system using open-source tools only.

## 🚀 Features

- **Audio Processing Pipeline**:
  - Multi-format audio ingestion (.mp3, .m4a, .flac, .wav)
  - High-quality transcription using Whisper/Faster-Whisper
  - Advanced speaker diarization with Pyannote-Audio
  - Dual-mode emotion detection (text + audio)
  - Semantic segmentation and topic modeling
  - Local LLM summarization via Ollama

- **RAG System**:
  - Vector storage with ChromaDB
  - Advanced query engine with metadata filtering
  - CLI and Streamlit web interfaces
  - Real-time search and analysis

## 📦 Installation

### 1. Clone and Setup Environment

```bash
git clone <repository>
cd podsmtih8
python -m venv pod
source pod/bin/activate  # On Windows: pod\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama

Download and install Ollama from: https://ollama.ai/

```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull mistral:7b
ollama pull llama3.2:3b
ollama pull qwen2.5:1.5b
```

### 3. Download Required Models

The pipeline will automatically download required models on first run:
- Whisper models (base, small, medium, large)
- Pyannote speaker diarization models
- Sentence transformers for embeddings
- Emotion detection models

## 🎯 Usage

### CLI Interface

```bash
# Process an audio file
python -m pipeline.pipeline_runner --audio_path "path/to/audio.mp3" --session_id "my_session"

# Query the processed data
python -m rag_system.interface_cli

# Example queries:
> what does the speaker say about krishna?
> filter emotion=joy
> summarize the main topics
> show segments from speaker 2 with confidence > 0.8
```

### Streamlit Web Interface

```bash
streamlit run rag_system/interface_streamlit.py
```

Features:
- Upload and process audio files
- Interactive query interface
- Filter by emotion, speaker, topic, confidence
- Real-time search results
- Visual analytics dashboard

### Programmatic Usage

```python
from pipeline.pipeline_runner import PipelineRunner
from rag_system.rag_core import RAGCore

# Process audio
runner = PipelineRunner()
session_id = runner.process_audio("path/to/audio.mp3")

# Query results
rag = RAGCore()
results = rag.query("What are the main topics discussed?", 
                   filters={"emotion": "joy", "confidence_min": 0.7})
```

## 🏗️ Architecture

### Pipeline Modules (`/pipeline/`)

- **audio_ingestion.py**: Audio format conversion and preprocessing
- **transcription.py**: Speech-to-text using Whisper
- **diarization.py**: Speaker separation using Pyannote
- **emotion_detection.py**: Text and audio emotion analysis
- **semantic_segmentation.py**: Topic modeling and text segmentation
- **summarization.py**: LLM-based content summarization
- **pipeline_runner.py**: Orchestrates the entire pipeline

### RAG System (`/rag_system/`)

- **vector_store.py**: ChromaDB integration for embeddings
- **query_engine.py**: Search and retrieval engine
- **rag_core.py**: Core RAG functionality
- **interface_cli.py**: Command-line interface
- **interface_streamlit.py**: Web interface

### Utilities (`/utils/`)

- **logger.py**: Centralized logging
- **file_utils.py**: File operations and caching

## 📊 Output Format

The pipeline generates a comprehensive analysis report:

```json
{
  "comprehensive_analysis": {
    "metadata": {
      "session_id": "session_20240117_143022",
      "audio_file": "podcast.mp3",
      "duration": 3600.5,
      "processing_time": 245.3
    },
    "block_wise_analysis": [
      {
        "block_id": 1,
        "block_summary": {
          "text": "Discussion about artificial intelligence...",
          "main_topics": ["AI", "Technology", "Future"],
          "key_speakers": ["Speaker 1", "Speaker 2"]
        },
        "original_segments": [
          {
            "segment_id": 8,
            "text": "This is a segment about AI development.",
            "speaker": "Speaker 1",
            "start_time": 125.3,
            "end_time": 138.7,
            "emotions": {
              "combined_emotion": {
                "emotion": "joy",
                "confidence": 0.92
              }
            },
            "topics": ["AI", "Technology"]
          }
        ]
      }
    ]
  }
}
```

## ⚙️ Configuration

### Audio Processing Settings

- **Sample Rate**: 16kHz (optimized for speech)
- **Channels**: Mono (automatic conversion)
- **Format**: 16-bit PCM WAV (in-memory processing)

### Model Settings

- **Whisper**: Configurable model size (base/small/medium/large)
- **Diarization**: Pyannote 3.1+ with embedding clustering
- **Emotion**: DistilRoBERTa for text, Wav2Vec2 for audio
- **Embeddings**: all-MiniLM-L6-v2 (384-dim, fast inference)

## 🔧 Advanced Features

### Memory Efficiency
- Streaming audio processing
- In-memory intermediate results
- Chunked processing for large files
- Automatic cleanup of temporary data

### Real-time Processing
- Progressive result updates
- Session-based resumable processing
- Background processing support
- Live progress tracking

### Extensibility
- Plugin architecture for new emotions
- Custom topic modeling algorithms
- Configurable LLM backends
- API endpoints for integration

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Ensure Ollama is running
   ollama serve
   # Check if models are available
   ollama list
   ```

2. **Audio Format Issues**:
   - Install FFmpeg: `conda install ffmpeg` or system package manager
   - Supported formats: MP3, M4A, FLAC, WAV, OGG

3. **Memory Issues**:
   - Reduce Whisper model size: use 'base' or 'small'
   - Process shorter audio segments
   - Increase system swap space

### Performance Optimization

- **GPU Acceleration**: Automatic CUDA detection for Whisper and emotion models
- **CPU Optimization**: Multi-threading for parallel processing
- **Memory Management**: Automatic cleanup and garbage collection

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check troubleshooting guide
- Review example notebooks in `/examples/`
