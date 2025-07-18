# 🎙️ Smart Audio RAG (Retrieval-Augmented Generation) System

An intelligent system that processes podcast transcripts and audio analysis data to create a powerful query-response interface with conversation memory, multi-modal search, and advanced feedback collection.

## ✨ Features

### 🔍 Multi-Modal Search Capabilities
- **Content Search**: Natural language queries across transcript content
- **Speaker-Specific Queries**: "What did Speaker_01 say about technology?"
- **Emotional Analysis**: "When was the conversation most excited?"
- **Temporal Queries**: "What happened around 5 minutes in?"
- **Analytical Queries**: "Summarize the main themes discussed"

### 🧠 Intelligent Query Processing
- **Intent Recognition**: Automatically understands query types and intentions
- **Entity Extraction**: Identifies speakers, emotions, topics, and time references
- **Context-Aware Responses**: Maintains conversation context for follow-up questions
- **Multi-Dimensional Ranking**: Advanced relevance scoring with multiple factors

### 💾 Conversation Memory
- **Persistent Storage**: SQLite-based conversation history and user preferences
- **Context Management**: Maintains conversation state and recent query context
- **User Learning**: Adapts to user preferences and query patterns
- **Session Management**: Supports multiple concurrent users and conversations

### 📊 Advanced Feedback System
- **Variable Timing**: Immediate, delayed, and session-end feedback collection
- **Pattern Learning**: Analyzes feedback to improve system performance
- **Adaptive Prompting**: Adjusts feedback frequency based on user satisfaction
- **Multiple Feedback Types**: Satisfaction, relevance, completeness, accuracy

### 🗄️ Hierarchical Vector Database
- **Multi-Collection Architecture**: Separate collections for episodes, blocks, segments, speakers, emotions
- **ChromaDB Backend**: Persistent vector storage with metadata filtering
- **Embedding Strategy**: SentenceTransformers for local, fast embeddings
- **Performance Optimization**: Efficient similarity search with caching

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd podsmtih8
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place your podcast session data in `output/sessions/`
   - Each session should have `final_report.json` and `summarization.json`
   - Example structure:
     ```
     output/sessions/
     ├── session_001/
     │   ├── final_report.json
     │   └── summarization.json
     └── session_002/
         ├── final_report.json
         └── summarization.json
     ```

### Running the System

#### Interactive Mode (Recommended)
```bash
python main.py interactive
```

#### Single Query Mode
```bash
python main.py query "What are the main topics discussed?"
```

#### With Custom Parameters
```bash
python main.py interactive --data-dir custom_data --session-dir my_sessions --user-id user123
```

## 🎯 Usage Examples

### Basic Queries
```
# Content search
"What topics were discussed in the podcast?"

# Speaker-specific
"What did Speaker_01 say about artificial intelligence?"

# Emotional context
"Show me moments when speakers were excited or enthusiastic"

# Time-based
"What happened in the first 10 minutes?"
"Tell me about the conversation around 15:30"

# Analytical
"Summarize the key themes and insights"
"Compare what different speakers said about technology"
```

### CLI Commands
```
/help          - Show available commands
/stats         - Display system statistics
/sessions      - List loaded sessions
/config        - Show current configuration
/feedback      - View feedback analytics
/conversation  - Show conversation history
/debug         - Toggle debug mode
/quit          - Exit the system
```

## 🏗️ Architecture

### Core Components

#### 1. Vector Database (`SmartVectorDB`)
- **ChromaDB Integration**: Persistent vector storage
- **Multi-Collection Design**: 
  - Episodes: Full episode metadata and summaries
  - Blocks: Thematic content blocks
  - Segments: Individual conversation segments
  - Speakers: Speaker-specific content and statistics
  - Emotions: Emotional content and analysis
- **Embedding Strategy**: SentenceTransformers (all-MiniLM-L6-v2)
- **Metadata Filtering**: Advanced filtering by time, speaker, emotion, confidence

#### 2. Query Processor (`SmartQueryProcessor`)
- **Intent Classification**: Automatic query type detection
- **Entity Extraction**: NLP-based entity identification
- **Context Integration**: Conversation memory integration
- **Result Ranking**: Multi-factor relevance scoring
- **Response Generation**: Context-aware response formatting

#### 3. Conversation Memory (`ConversationMemory`)
- **SQLite Storage**: Persistent conversation and user data
- **Context Management**: Recent query tracking and topic continuity
- **User Preferences**: Learned patterns and personalization
- **Session Support**: Multi-user conversation management

#### 4. Feedback System (`FeedbackSystem`)
- **Adaptive Collection**: Variable timing based on user behavior
- **Pattern Analysis**: Machine learning from user feedback
- **Multiple Types**: Satisfaction, relevance, completeness, accuracy
- **Fatigue Prevention**: Smart frequency adjustment

### Data Flow
```
User Query → Intent Analysis → Entity Extraction → Context Integration 
           ↓
Vector Search → Result Ranking → Response Generation → Feedback Collection
           ↓
Conversation Memory Update → User Preference Learning
```
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
