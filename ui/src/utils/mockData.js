// Mock data for quotes
export const mockQuotes = [
  {
    text: "The key to success is not just working hard, but working smart and understanding your audience.",
    speaker: "Speaker 1",
    emotion: "joy",
    session_id: "session_test1",
    start_time: 120,
    end_time: 127,
    duration: 7,
    confidence: 0.89,
    context: "Discussion about business strategies and market understanding...",
  },
  {
    text: "I think we need to be more careful about how we approach this problem.",
    speaker: "Speaker 2", 
    emotion: "neutral",
    session_id: "session_test1",
    start_time: 245,
    end_time: 250,
    duration: 5,
    confidence: 0.76,
    context: "Team discussion about project challenges and risk management...",
  },
];

// Mock data for analytics
export const mockAnalytics = {
  total_sessions: 12,
  total_duration: 14400, // 4 hours in seconds
  average_duration: 1200, // 20 minutes
  unique_speakers: 8,
  emotions: {
    joy: 45,
    neutral: 38,
    sadness: 12,
    anger: 8,
    fear: 5,
    surprise: 15,
  },
  speakers: {
    "Speaker 1": {
      total_duration: 3600,
      segment_count: 25,
    },
    "Speaker 2": {
      total_duration: 2400,
      segment_count: 18,
    },
    "Speaker 3": {
      total_duration: 1800,
      segment_count: 12,
    },
  },
  processing_times: [
    { date: "2024-01-01", processing_time: 180, file_size_mb: 25 },
    { date: "2024-01-02", processing_time: 220, file_size_mb: 30 },
    { date: "2024-01-03", processing_time: 165, file_size_mb: 22 },
    { date: "2024-01-04", processing_time: 195, file_size_mb: 28 },
    { date: "2024-01-05", processing_time: 175, file_size_mb: 24 },
  ],
};

// Mock data for settings
export const mockSettings = {
  pipeline: {
    audio_quality: "medium",
    max_concurrent: 2,
    enable_diarization: true,
    enable_emotion_detection: true,
    auto_cleanup: false,
    temp_dir: "/tmp/podcast_processing",
  },
  rag: {
    vector_store: "chroma",
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: 512,
    chunk_overlap: 50,
    top_k: 5,
    vector_store_path: "./output/chroma_db",
  },
  security: {
    require_authentication: false,
    enable_rate_limiting: true,
    api_keys: ["demo_key_1234", "test_key_5678"],
  },
  storage: {
    output_dir: "./output",
    logs_dir: "./logs",
    max_size_gb: 10,
    auto_backup: false,
  },
  performance: {
    enable_caching: true,
    enable_gpu: false,
    memory_limit_gb: 8,
  },
};
