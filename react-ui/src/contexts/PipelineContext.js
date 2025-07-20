import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { io } from 'socket.io-client';

const PipelineContext = createContext();

const initialState = {
  sessions: [],
  activeSessions: [],
  completedSessions: [],
  currentProcessing: null,
  uploadQueue: [],
  socket: null,
  connected: false,
  processingStats: {
    totalProcessed: 0,
    averageTime: 0,
    successRate: 100,
  },
};

const pipelineReducer = (state, action) => {
  switch (action.type) {
    case 'SET_SOCKET':
      return { ...state, socket: action.payload };
    
    case 'SET_CONNECTED':
      return { ...state, connected: action.payload };
    
    case 'ADD_TO_QUEUE':
      return {
        ...state,
        uploadQueue: [...state.uploadQueue, action.payload],
      };
    
    case 'REMOVE_FROM_QUEUE':
      return {
        ...state,
        uploadQueue: state.uploadQueue.filter(item => item.id !== action.payload),
      };
    
    case 'START_PROCESSING':
      return {
        ...state,
        currentProcessing: action.payload,
        activeSessions: [...state.activeSessions, action.payload],
      };
    
    case 'UPDATE_PROCESSING':
      return {
        ...state,
        currentProcessing: action.payload.sessionId === state.currentProcessing?.sessionId 
          ? { ...state.currentProcessing, ...action.payload } 
          : state.currentProcessing,
        activeSessions: state.activeSessions.map(session =>
          session.sessionId === action.payload.sessionId
            ? { ...session, ...action.payload }
            : session
        ),
      };
    
    case 'COMPLETE_PROCESSING':
      return {
        ...state,
        currentProcessing: null,
        activeSessions: state.activeSessions.filter(s => s.sessionId !== action.payload.sessionId),
        completedSessions: [...state.completedSessions, action.payload],
      };
    
    case 'SET_SESSIONS':
      return { ...state, sessions: action.payload };
    
    case 'UPDATE_STATS':
      return { ...state, processingStats: { ...state.processingStats, ...action.payload } };
    
    default:
      return state;
  }
};

export const PipelineProvider = ({ children }) => {
  const [state, dispatch] = useReducer(pipelineReducer, initialState);

  useEffect(() => {
    // Initialize socket connection
    const socket = io('http://localhost:8000', {
      transports: ['websocket', 'polling'],
    });

    dispatch({ type: 'SET_SOCKET', payload: socket });

    socket.on('connect', () => {
      console.log('Connected to pipeline server');
      dispatch({ type: 'SET_CONNECTED', payload: true });
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from pipeline server');
      dispatch({ type: 'SET_CONNECTED', payload: false });
    });

    socket.on('processing_started', (data) => {
      dispatch({ type: 'START_PROCESSING', payload: data });
    });

    socket.on('processing_update', (data) => {
      dispatch({ type: 'UPDATE_PROCESSING', payload: data });
    });

    socket.on('processing_completed', (data) => {
      dispatch({ type: 'COMPLETE_PROCESSING', payload: data });
    });

    socket.on('processing_error', (data) => {
      console.error('Processing error:', data);
      dispatch({ type: 'COMPLETE_PROCESSING', payload: { ...data, error: true } });
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const uploadFiles = async (files) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        return result;
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  };

  const startProcessing = async (sessionId, config = {}) => {
    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sessionId, config }),
      });
      
      if (response.ok) {
        const result = await response.json();
        return result;
      } else {
        throw new Error('Processing start failed');
      }
    } catch (error) {
      console.error('Processing start error:', error);
      throw error;
    }
  };

  const resumeSession = async (sessionId) => {
    try {
      const response = await fetch(`/api/resume/${sessionId}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        return result;
      } else {
        throw new Error('Session resume failed');
      }
    } catch (error) {
      console.error('Session resume error:', error);
      throw error;
    }
  };

  const getSessions = async () => {
    try {
      const response = await fetch('/api/sessions');
      if (response.ok) {
        const sessions = await response.json();
        dispatch({ type: 'SET_SESSIONS', payload: sessions });
        return sessions;
      }
    } catch (error) {
      console.error('Get sessions error:', error);
    }
    return [];
  };

  const getSessionDetails = async (sessionId) => {
    try {
      const response = await fetch(`/api/sessions/${sessionId}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Get session details error:', error);
    }
    return null;
  };

  const contextValue = {
    ...state,
    dispatch,
    uploadFiles,
    startProcessing,
    resumeSession,
    getSessions,
    getSessionDetails,
  };

  return (
    <PipelineContext.Provider value={contextValue}>
      {children}
    </PipelineContext.Provider>
  );
};

export const usePipeline = () => {
  const context = useContext(PipelineContext);
  if (!context) {
    throw new Error('usePipeline must be used within a PipelineProvider');
  }
  return context;
};
