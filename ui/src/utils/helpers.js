// Time utilities
export const formatDuration = (seconds) => {
  if (!seconds || seconds < 0) return '0:00';
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

export const formatTimestamp = (timestamp) => {
  if (!timestamp) return '';
  
  const date = new Date(timestamp);
  return date.toLocaleString();
};

export const formatRelativeTime = (timestamp) => {
  if (!timestamp) return '';
  
  const now = new Date();
  const date = new Date(timestamp);
  const diffMs = now - date;
  const diffMinutes = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  
  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes} minutes ago`;
  if (diffHours < 24) return `${diffHours} hours ago`;
  if (diffDays < 7) return `${diffDays} days ago`;
  
  return date.toLocaleDateString();
};

// File utilities
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const getFileExtension = (filename) => {
  return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
};

export const isAudioFile = (filename) => {
  const audioExtensions = ['mp3', 'm4a', 'wav', 'flac', 'ogg', 'aac', 'mp4', 'wma'];
  const extension = getFileExtension(filename).toLowerCase();
  return audioExtensions.includes(extension);
};

// Processing utilities
export const getProcessingStepName = (step) => {
  const stepNames = {
    'audio_ingestion': 'Audio Ingestion',
    'transcription': 'Transcription',
    'diarization': 'Speaker Diarization',
    'segment_enrichment': 'Segment Enrichment',
    'emotion_detection': 'Emotion Detection',
    'semantic_segmentation': 'Semantic Segmentation',
    'summarization': 'Summarization',
    'pipeline_complete': 'Complete',
  };
  return stepNames[step] || step;
};

export const getProcessingStepIcon = (step) => {
  const stepIcons = {
    'audio_ingestion': '🎵',
    'transcription': '📝',
    'diarization': '👥',
    'segment_enrichment': '🔗',
    'emotion_detection': '😊',
    'semantic_segmentation': '📊',
    'summarization': '📋',
    'pipeline_complete': '✅',
  };
  return stepIcons[step] || '⚙️';
};

export const calculateProgress = (completedSteps, totalSteps = 8) => {
  return Math.round((completedSteps.length / totalSteps) * 100);
};

// Data utilities
export const truncateText = (text, maxLength = 100) => {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

export const highlightText = (text, searchTerm) => {
  if (!searchTerm || !text) return text;
  
  const regex = new RegExp(`(${searchTerm})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
};

export const extractKeywords = (text, count = 5) => {
  if (!text) return [];
  
  // Simple keyword extraction (can be improved with NLP)
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 3)
    .filter(word => !['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'what', 'were'].includes(word));
  
  const frequency = {};
  words.forEach(word => {
    frequency[word] = (frequency[word] || 0) + 1;
  });
  
  return Object.entries(frequency)
    .sort(([,a], [,b]) => b - a)
    .slice(0, count)
    .map(([word]) => word);
};

// Validation utilities
export const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
};

export const validateFileName = (filename) => {
  const invalidChars = /[<>:"/\\|?*]/g;
  return !invalidChars.test(filename);
};

// Session utilities
export const generateSessionId = () => {
  const timestamp = new Date().toISOString().replace(/[-:]/g, '').split('.')[0];
  const random = Math.random().toString(36).substring(2, 8);
  return `session_${timestamp}_${random}`;
};

export const parseSessionId = (sessionId) => {
  const parts = sessionId.split('_');
  if (parts.length >= 3) {
    const timestamp = parts[1];
    const year = timestamp.substring(0, 4);
    const month = timestamp.substring(4, 6);
    const day = timestamp.substring(6, 8);
    const hour = timestamp.substring(9, 11);
    const minute = timestamp.substring(11, 13);
    
    return {
      date: `${year}-${month}-${day}`,
      time: `${hour}:${minute}`,
      id: parts[2],
    };
  }
  return { date: '', time: '', id: sessionId };
};

// Color utilities
export const getEmotionColor = (emotion) => {
  const emotionColors = {
    'joy': '#4CAF50',
    'sadness': '#2196F3',
    'anger': '#F44336',
    'fear': '#9C27B0',
    'surprise': '#FF9800',
    'disgust': '#795548',
    'neutral': '#9E9E9E',
    'love': '#E91E63',
    'excitement': '#FFEB3B',
    'confusion': '#607D8B',
  };
  return emotionColors[emotion?.toLowerCase()] || '#9E9E9E';
};

export const getSpeakerColor = (speakerId) => {
  const colors = [
    '#1976D2', '#388E3C', '#F57C00', '#7B1FA2',
    '#C2185B', '#00796B', '#5D4037', '#455A64',
    '#E64A19', '#512DA8', '#689F38', '#FBC02D'
  ];
  
  if (typeof speakerId === 'string') {
    // Extract number from speaker ID (e.g., "Speaker_01" -> 1)
    const match = speakerId.match(/\d+/);
    const index = match ? parseInt(match[0]) : 0;
    return colors[index % colors.length];
  }
  
  return colors[speakerId % colors.length];
};

// Chart utilities
export const prepareChartData = (data, xKey, yKey) => {
  if (!Array.isArray(data)) return [];
  
  return data.map(item => ({
    [xKey]: item[xKey],
    [yKey]: item[yKey],
  }));
};

export const aggregateByTimeSlice = (segments, sliceDuration = 60) => {
  if (!Array.isArray(segments)) return [];
  
  const slices = {};
  
  segments.forEach(segment => {
    const startTime = segment.start_time || 0;
    const sliceIndex = Math.floor(startTime / sliceDuration);
    const sliceKey = `${sliceIndex * sliceDuration}s`;
    
    if (!slices[sliceKey]) {
      slices[sliceKey] = {
        time: sliceKey,
        segments: [],
        duration: 0,
        emotions: {},
        speakers: new Set(),
      };
    }
    
    slices[sliceKey].segments.push(segment);
    slices[sliceKey].duration += (segment.end_time || 0) - (segment.start_time || 0);
    
    if (segment.speaker) {
      slices[sliceKey].speakers.add(segment.speaker);
    }
    
    if (segment.emotion || segment.text_emotion?.emotion) {
      const emotion = segment.emotion || segment.text_emotion.emotion;
      slices[sliceKey].emotions[emotion] = (slices[sliceKey].emotions[emotion] || 0) + 1;
    }
  });
  
  return Object.values(slices).map(slice => ({
    ...slice,
    speakers: Array.from(slice.speakers),
    dominantEmotion: Object.entries(slice.emotions).sort(([,a], [,b]) => b - a)[0]?.[0] || 'neutral',
  }));
};

// Export utilities
export const downloadJSON = (data, filename) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const downloadCSV = (data, filename) => {
  if (!Array.isArray(data) || data.length === 0) return;
  
  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => headers.map(header => {
      const value = row[header];
      if (typeof value === 'string' && value.includes(',')) {
        return `"${value}"`;
      }
      return value;
    }).join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
