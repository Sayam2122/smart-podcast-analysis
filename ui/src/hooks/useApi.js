import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../services/apiClient';
import { mockQuotes, mockAnalytics, mockSettings } from '../utils/mockData';

// Sessions API hooks
export const useSessions = () => {
  return useQuery({
    queryKey: ['sessions'],
    queryFn: () => apiClient.get('/sessions'),
    staleTime: 30000, // 30 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
};

export const useSession = (sessionId) => {
  return useQuery({
    queryKey: ['sessions', sessionId],
    queryFn: () => apiClient.get(`/sessions/${sessionId}`),
    enabled: !!sessionId,
  });
};

// Processing API hooks
export const useProcessFile = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (formData) => apiClient.post('/pipeline/start', formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
};

export const useProcessingStatus = () => {
  return useQuery({
    queryKey: ['processing-status'],
    queryFn: () => apiClient.get('/pipeline/status'),
    refetchInterval: 2000, // Poll every 2 seconds
    retry: 2,
  });
};

// Chat API hooks
export const useChat = (sessionId, chatType = 'main') => {
  return useQuery({
    queryKey: ['chat', sessionId, chatType],
    queryFn: () => apiClient.get(`/chat/${sessionId}/${chatType}`),
    enabled: !!sessionId,
  });
};

export const useSendMessage = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ sessionId, chatType, message }) => 
      apiClient.post(`/chat/${sessionId}/${chatType}`, { message }),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ 
        queryKey: ['chat', variables.sessionId, variables.chatType] 
      });
    },
  });
};

// System status hooks
export const useSystemStatus = () => {
  return useQuery({
    queryKey: ['system-status'],
    queryFn: () => apiClient.get('/system/status'),
    refetchInterval: 5000, // Poll every 5 seconds
    retry: 2,
  });
};

// Analytics hooks
export const useAnalytics = (params = {}) => {
  return useQuery({
    queryKey: ['analytics', params],
    queryFn: async () => {
      // Mock implementation - in real app this would call the API
      return new Promise((resolve) => {
        setTimeout(() => resolve({ data: mockAnalytics }), 500);
      });
    },
    staleTime: 60000, // 1 minute
  });
};

// Quotes hooks
export const useQuotes = () => {
  const queryClient = useQueryClient();
  
  const extractQuotes = useMutation({
    mutationFn: async (params) => {
      // Mock implementation - in real app this would call the API
      return new Promise((resolve) => {
        setTimeout(() => {
          // Filter mock quotes based on params
          let filteredQuotes = [...mockQuotes];
          
          if (params.filters?.searchText) {
            filteredQuotes = filteredQuotes.filter(q => 
              q.text.toLowerCase().includes(params.filters.searchText.toLowerCase())
            );
          }
          
          if (params.filters?.emotion) {
            filteredQuotes = filteredQuotes.filter(q => q.emotion === params.filters.emotion);
          }
          
          if (params.filters?.speaker) {
            filteredQuotes = filteredQuotes.filter(q => q.speaker === params.filters.speaker);
          }
          
          resolve({ data: { quotes: filteredQuotes } });
        }, 1000);
      });
    },
  });

  return {
    extractQuotes: extractQuotes.mutateAsync,
    isExtracting: extractQuotes.isPending,
  };
};

// Settings hooks
export const useSettings = () => {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: ['settings'],
    queryFn: async () => {
      // Mock implementation - in real app this would call the API
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockSettings), 300);
      });
    },
  });

  const updateSettings = useMutation({
    mutationFn: async (newSettings) => {
      // Mock implementation - in real app this would call the API
      return new Promise((resolve) => {
        setTimeout(() => resolve(newSettings), 500);
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });

  const resetSettings = useMutation({
    mutationFn: async () => {
      // Mock implementation - in real app this would call the API
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockSettings), 500);
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });

  return {
    data: query.data,
    isLoading: query.isLoading,
    updateSettings: updateSettings.mutateAsync,
    resetSettings: resetSettings.mutateAsync,
  };
};

// Additional utility hooks for file uploads and processing
export const useUpload = () => {
  const queryClient = useQueryClient();
  const [uploadProgress, setUploadProgress] = useState(0);

  const uploadMutation = useMutation({
    mutationFn: async ({ files, config }) => {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });
      
      if (config) {
        formData.append('config', JSON.stringify(config));
      }

      return apiClient.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setUploadProgress(0);
    },
    onError: () => {
      setUploadProgress(0);
    },
  });

  return {
    upload: uploadMutation.mutateAsync,
    isUploading: uploadMutation.isPending,
    uploadProgress,
    error: uploadMutation.error,
  };
};

// Health check hook
export const useHealthCheck = () => {
  const [isHealthy, setIsHealthy] = useState(false);
  const [lastCheck, setLastCheck] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.get('/health');
        setIsHealthy(true);
        setLastCheck(new Date());
      } catch (error) {
        setIsHealthy(false);
        setLastCheck(new Date());
        console.error('Health check failed:', error);
      }
    };

    // Initial check
    checkHealth();

    // Check every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, []);

  return { isHealthy, lastCheck };
};

// System metrics hook
export const useSystemMetrics = () => {
  return useQuery({
    queryKey: ['system-metrics'],
    queryFn: () => apiClient.get('/system/info'),
    refetchInterval: 10000, // Poll every 10 seconds
    retry: 2,
    staleTime: 5000, // Consider data stale after 5 seconds
  });
};

// Watch folder hook
export const useWatchFolder = () => {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: ['watchfolder-files'],
    queryFn: () => apiClient.get('/files/watchfolder'),
    refetchInterval: 5000, // Poll every 5 seconds for file changes
    retry: 2,
  });

  const deleteFile = useMutation({
    mutationFn: (filename) => apiClient.delete(`/files/${filename}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['watchfolder-files'] });
    },
  });

  return {
    ...query,
    deleteFile: deleteFile.mutateAsync,
    isDeletingFile: deleteFile.isPending,
    deleteError: deleteFile.error,
  };
};

// Processing hook (comprehensive processing management)
export const useProcessing = () => {
  const queryClient = useQueryClient();
  
  // Get current processing status
  const statusQuery = useQuery({
    queryKey: ['processing-status'],
    queryFn: () => apiClient.get('/pipeline/status'),
    refetchInterval: 2000, // Poll every 2 seconds during processing
    retry: 2,
  });

  // Start processing mutation
  const startProcessing = useMutation({
    mutationFn: ({ files, config }) => {
      const payload = {
        files: files.map(f => typeof f === 'string' ? f : f.name),
        config: config || {}
      };
      return apiClient.post('/pipeline/start', payload);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['processing-status'] });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });

  // Stop processing mutation
  const stopProcessing = useMutation({
    mutationFn: () => apiClient.post('/pipeline/stop'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['processing-status'] });
    },
  });

  // Resume session mutation
  const resumeSession = useMutation({
    mutationFn: (sessionId) => apiClient.post(`/pipeline/resume/${sessionId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['processing-status'] });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });

  return {
    // Status data
    status: statusQuery.data,
    isLoadingStatus: statusQuery.isLoading,
    statusError: statusQuery.error,
    
    // Processing state helpers
    isProcessing: statusQuery.data?.is_processing || false,
    currentStep: statusQuery.data?.current_step,
    progress: statusQuery.data?.progress || 0,
    currentFile: statusQuery.data?.current_file,
    currentSession: statusQuery.data?.current_session,
    error: statusQuery.data?.error,
    
    // Actions
    startProcessing: startProcessing.mutateAsync,
    stopProcessing: stopProcessing.mutateAsync,
    resumeSession: resumeSession.mutateAsync,
    
    // Action states
    isStarting: startProcessing.isPending,
    isStopping: stopProcessing.isPending,
    isResuming: resumeSession.isPending,
    
    // Errors
    startError: startProcessing.error,
    stopError: stopProcessing.error,
    resumeError: resumeSession.error,
  };
};
