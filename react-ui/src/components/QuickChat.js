import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  Chip,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Avatar,
  Divider,
} from '@mui/material';
import {
  Send,
  Person,
  SmartToy,
  Refresh,
} from '@mui/icons-material';

const QuickChat = ({ onNotification }) => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [episodes, setEpisodes] = useState([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState([]);
  const [loadingEpisodes, setLoadingEpisodes] = useState(true);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    loadEpisodes();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadEpisodes = async () => {
    try {
      setLoadingEpisodes(true);
      const response = await fetch('/api/rag/episodes');
      if (response.ok) {
        const data = await response.json();
        setEpisodes(data.episodes);
      } else {
        throw new Error('Failed to load episodes');
      }
    } catch (error) {
      console.error('Error loading episodes:', error);
      onNotification('Failed to load episodes', 'error');
    } finally {
      setLoadingEpisodes(false);
    }
  };

  const handleSendMessage = async () => {
    if (!message.trim()) return;
    
    if (selectedEpisodes.length === 0) {
      onNotification('Please select at least one episode to chat with', 'warning');
      return;
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setLoading(true);

    try {
      const response = await fetch('/api/rag/quick-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          episodes: selectedEpisodes,
          conversation_type: 'quick',
        }),
      });

      if (response.ok) {
        const data = await response.json();
        
        const botMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: data.response,
          sources: data.sources,
          confidence: data.confidence,
          timestamp: new Date(),
        };

        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      onNotification('Failed to send message', 'error');
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error processing your message. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  if (loadingEpisodes) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading episodes...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', height: 'calc(100vh - 200px)' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Quick Chat
      </Typography>
      
      <Grid container spacing={3} sx={{ height: '100%' }}>
        {/* Episode Selection Panel */}
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Episodes</Typography>
                <Button size="small" startIcon={<Refresh />} onClick={loadEpisodes}>
                  Refresh
                </Button>
              </Box>
              
              {episodes.length === 0 ? (
                <Alert severity="info">
                  No processed episodes found. Please process some audio files first.
                </Alert>
              ) : (
                <>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Selected Episodes</InputLabel>
                    <Select
                      multiple
                      value={selectedEpisodes}
                      onChange={(e) => setSelectedEpisodes(e.target.value)}
                      label="Selected Episodes"
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {episodes.map((episode) => (
                        <MenuItem key={episode.id} value={episode.id}>
                          {episode.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={() => setSelectedEpisodes(episodes.map(ep => ep.id))}
                    sx={{ mb: 1 }}
                  >
                    Select All
                  </Button>
                  
                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={() => setSelectedEpisodes([])}
                  >
                    Clear Selection
                  </Button>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Available Episodes:
                    </Typography>
                    <List dense>
                      {episodes.map((episode) => (
                        <ListItem 
                          key={episode.id}
                          sx={{ 
                            pl: 0,
                            backgroundColor: selectedEpisodes.includes(episode.id) ? 'action.selected' : 'transparent',
                            borderRadius: 1,
                            mb: 0.5,
                          }}
                        >
                          <ListItemText
                            primary={episode.name}
                            secondary={`${episode.segments || 0} segments`}
                            primaryTypographyProps={{ variant: 'body2' }}
                            secondaryTypographyProps={{ variant: 'caption' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Chat Interface */}
        <Grid item xs={12} md={9}>
          <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Chat Header */}
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">
                  Quick Chat {selectedEpisodes.length > 0 && `(${selectedEpisodes.length} episodes selected)`}
                </Typography>
                <Button size="small" onClick={clearChat}>
                  Clear Chat
                </Button>
              </Box>
            </Box>

            {/* Messages Area */}
            <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
              {messages.length === 0 ? (
                <Box sx={{ textAlign: 'center', mt: 4 }}>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Start a conversation
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Select episodes and ask questions about the podcast content
                  </Typography>
                </Box>
              ) : (
                messages.map((msg) => (
                  <Box key={msg.id} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                      <Avatar sx={{ 
                        bgcolor: msg.type === 'user' ? 'primary.main' : 'secondary.main',
                        width: 32, 
                        height: 32 
                      }}>
                        {msg.type === 'user' ? <Person /> : <SmartToy />}
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Typography variant="subtitle2">
                            {msg.type === 'user' ? 'You' : 'Assistant'}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {formatTimestamp(msg.timestamp)}
                          </Typography>
                          {msg.confidence && (
                            <Chip 
                              label={`${Math.round(msg.confidence * 100)}% confident`} 
                              size="small" 
                              color={msg.confidence > 0.8 ? 'success' : msg.confidence > 0.6 ? 'warning' : 'default'}
                            />
                          )}
                        </Box>
                        <Paper 
                          sx={{ 
                            p: 1.5, 
                            bgcolor: msg.type === 'user' ? 'primary.light' : 'background.default',
                            color: msg.type === 'user' ? 'primary.contrastText' : 'text.primary',
                          }}
                        >
                          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                            {msg.content}
                          </Typography>
                        </Paper>
                        
                        {msg.sources && msg.sources.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Sources:
                            </Typography>
                            {msg.sources.map((source, index) => (
                              <Chip
                                key={index}
                                label={`${source.episode} (${Math.round(source.score * 100)}%)`}
                                size="small"
                                variant="outlined"
                                sx={{ ml: 0.5, mt: 0.5 }}
                              />
                            ))}
                          </Box>
                        )}
                      </Box>
                    </Box>
                  </Box>
                ))
              )}
              
              {loading && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
                  <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32 }}>
                    <SmartToy />
                  </Avatar>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CircularProgress size={16} />
                    <Typography variant="body2" color="text.secondary">
                      Thinking...
                    </Typography>
                  </Box>
                </Box>
              )}
              
              <div ref={messagesEndRef} />
            </Box>

            <Divider />

            {/* Input Area */}
            <Box sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  multiline
                  maxRows={4}
                  placeholder="Ask about the podcast content..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading || selectedEpisodes.length === 0}
                />
                <Button
                  variant="contained"
                  endIcon={<Send />}
                  onClick={handleSendMessage}
                  disabled={loading || !message.trim() || selectedEpisodes.length === 0}
                  sx={{ minWidth: 100 }}
                >
                  Send
                </Button>
              </Box>
              {selectedEpisodes.length === 0 && (
                <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
                  Please select episodes to start chatting
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default QuickChat;
