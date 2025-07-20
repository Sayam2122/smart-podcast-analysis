import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API methods
export const apiService = {
  // System status
  getSystemStatus: () => api.get('/api/system/status'),
  
  // File upload and management
  uploadFiles: (files, onProgress) => {
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append(`files`, file);
    });
    
    return api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: onProgress,
    });
  },
  
  getWatchFolder: () => api.get('/api/files/watchfolder'),
  deleteFile: (filename) => api.delete(`/api/files/${filename}`),
  
  // Pipeline processing
  startProcessing: (files, config) => api.post('/api/pipeline/start', { files, config }),
  getProcessingStatus: () => api.get('/api/pipeline/status'),
  stopProcessing: () => api.post('/api/pipeline/stop'),
  resumeSession: (sessionId) => api.post(`/api/pipeline/resume/${sessionId}`),
  
  // Sessions
  getSessions: () => api.get('/api/sessions'),
  getSession: (sessionId) => api.get(`/api/sessions/${sessionId}`),
  deleteSession: (sessionId) => api.delete(`/api/sessions/${sessionId}`),
  getSessionLogs: (sessionId) => api.get(`/api/sessions/${sessionId}/logs`),
  
  // RAG and Chat
  initializeRAG: () => api.post('/api/rag/initialize'),
  getRAGStatus: () => api.get('/api/rag/status'),
  
  // Quick Chat
  quickChat: (message, sessionIds = []) => 
    api.post('/api/chat/quick', { message, session_ids: sessionIds }),
  
  // Main Chat
  mainChat: (message, conversationId, sessionIds = []) =>
    api.post('/api/chat/main', { 
      message, 
      conversation_id: conversationId,
      session_ids: sessionIds 
    }),
  
  getConversations: () => api.get('/api/chat/conversations'),
  getConversation: (conversationId) => api.get(`/api/chat/conversations/${conversationId}`),
  deleteConversation: (conversationId) => api.delete(`/api/chat/conversations/${conversationId}`),
  
  // Quote extraction
  extractQuotes: (sessionIds, filters) =>
    api.post('/api/quotes/extract', { session_ids: sessionIds, filters }),
  
  getQuotes: (sessionId) => api.get(`/api/quotes/${sessionId}`),
  
  // Analytics
  getAnalytics: (sessionIds) => api.post('/api/analytics', { session_ids: sessionIds }),
  getSystemMetrics: () => api.get('/api/analytics/system'),
  
  // Content generation
  generateContent: (type, sessionIds, config) =>
    api.post('/api/content/generate', { type, session_ids: sessionIds, config }),
  
  // Settings
  getSettings: () => api.get('/api/settings'),
  updateSettings: (settings) => api.post('/api/settings', settings),
  
  // System operations
  healthCheck: () => api.get('/api/health'),
  getSystemInfo: () => api.get('/api/system/info'),
};

export default api;
