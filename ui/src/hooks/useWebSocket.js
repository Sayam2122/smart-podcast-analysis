import { useState, useEffect } from 'react';
import wsService from '../services/websocket';

export const useWebSocket = () => {
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);

  useEffect(() => {
    // Connect when component mounts
    wsService.connect();

    // Listen for connection status changes
    const handleConnectionStatus = (status) => {
      setConnected(status.connected);
      if (status.failed || status.reason) {
        setConnectionError(status.reason || 'Connection failed');
      } else {
        setConnectionError(null);
      }
    };

    wsService.on('connection_status', handleConnectionStatus);

    // Cleanup on unmount
    return () => {
      wsService.off('connection_status', handleConnectionStatus);
      wsService.disconnect();
    };
  }, []);

  return {
    connected,
    connectionError,
    send: wsService.send.bind(wsService),
    on: wsService.on.bind(wsService),
    off: wsService.off.bind(wsService),
  };
};

export const useProcessingStatus = () => {
  const [status, setStatus] = useState({
    isProcessing: false,
    currentStep: null,
    progress: 0,
    error: null,
    completedSteps: [],
    currentFile: null,
    sessionId: null,
  });

  useEffect(() => {
    const handleProcessingStatus = (data) => {
      setStatus(prev => ({
        ...prev,
        isProcessing: data.status === 'started' || data.status === 'progress',
        currentStep: data.current_step || prev.currentStep,
        progress: data.progress || prev.progress,
        error: data.status === 'error' ? data.error : null,
        completedSteps: data.completed_steps || prev.completedSteps,
        currentFile: data.current_file || prev.currentFile,
        sessionId: data.session_id || prev.sessionId,
      }));
    };

    const handleStepCompleted = (data) => {
      setStatus(prev => ({
        ...prev,
        completedSteps: [...prev.completedSteps, data.step],
        progress: data.progress || prev.progress,
      }));
    };

    wsService.on('processing_status', handleProcessingStatus);
    wsService.on('step_completed', handleStepCompleted);

    return () => {
      wsService.off('processing_status', handleProcessingStatus);
      wsService.off('step_completed', handleStepCompleted);
    };
  }, []);

  return status;
};

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleChatResponse = (data) => {
      setMessages(prev => [...prev, {
        id: data.id || Date.now(),
        type: 'assistant',
        content: data.response,
        timestamp: new Date(),
        metadata: data.metadata,
      }]);
      setIsLoading(false);
      setError(null);
    };

    const handleChatError = (data) => {
      setError(data.error);
      setIsLoading(false);
    };

    wsService.on('chat_response', handleChatResponse);
    wsService.on('chat_error', handleChatError);

    return () => {
      wsService.off('chat_response', handleChatResponse);
      wsService.off('chat_error', handleChatError);
    };
  }, []);

  const sendMessage = (message, type = 'quick', sessionIds = []) => {
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    wsService.sendChatMessage(message, type, sessionIds);
  };

  const clearMessages = () => {
    setMessages([]);
    setError(null);
  };

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
  };
};

export const useSystemStatus = () => {
  const [systemStatus, setSystemStatus] = useState({
    healthy: false,
    services: {},
    resources: {},
  });

  useEffect(() => {
    const handleSystemStatus = (data) => {
      setSystemStatus(data);
    };

    wsService.on('system_status', handleSystemStatus);

    return () => {
      wsService.off('system_status', handleSystemStatus);
    };
  }, []);

  return systemStatus;
};
