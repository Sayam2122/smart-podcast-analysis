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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
  Slider,
  Tab,
  Tabs,
  IconButton,
} from '@mui/material';
import {
  Send,
  Person,
  SmartToy,
  Refresh,
  Settings,
  History,
  BookmarkAdd,
  Share,
  Download,
} from '@mui/icons-material';

const MainChat = ({ onNotification }) => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [episodes, setEpisodes] = useState([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState([]);
  const [loadingEpisodes, setLoadingEpisodes] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState({
    contextLength: 5,
    temperature: 0.7,
    maxTokens: 500,
    includeSourceContext: true,
    enableFeedback: true,
    conversationMode: true,
  });
  const messagesEndRef = useRef(null);

  useEffect(() => {
    loadEpisodes();
    loadConversations();
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

  const loadConversations = async () => {
    try {
      const response = await fetch('/api/rag/conversations');
      if (response.ok) {
        const data = await response.json();
        setConversations(data.conversations);
      }
    } catch (error) {
      console.error('Error loading conversations:', error);
    }
  };

  const startNewConversation = async () => {
    try {
      const response = await fetch('/api/rag/conversations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: `Conversation ${new Date().toLocaleString()}`,
          episodes: selectedEpisodes,
          settings: settings,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentConversationId(data.conversationId);
        setMessages([]);
        loadConversations();
        onNotification('New conversation started', 'success');
      }
    } catch (error) {
      console.error('Error starting conversation:', error);
      onNotification('Failed to start new conversation', 'error');
    }
  };

  const loadConversation = async (conversationId) => {
    try {
      const response = await fetch(`/api/rag/conversations/${conversationId}`);
      if (response.ok) {
        const data = await response.json();
        setCurrentConversationId(conversationId);
        setMessages(data.messages);
        setSelectedEpisodes(data.episodes);
        setSettings(data.settings || settings);
      }
    } catch (error) {
      console.error('Error loading conversation:', error);
      onNotification('Failed to load conversation', 'error');
    }
  };

  const handleSendMessage = async () => {
    if (!message.trim()) return;
    
    if (selectedEpisodes.length === 0) {
      onNotification('Please select at least one episode to chat with', 'warning');
      return;
    }

    // Start new conversation if none exists
    if (!currentConversationId) {
      await startNewConversation();
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
      const response = await fetch('/api/rag/main-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          episodes: selectedEpisodes,
          conversationId: currentConversationId,
          settings: settings,
          conversation_type: 'main',
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
          context: data.context,
          suggestions: data.suggestions,
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
    setCurrentConversationId(null);
  };

  const saveConversation = async () => {
    if (!currentConversationId) return;
    
    try {
      const response = await fetch(`/api/rag/conversations/${currentConversationId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages,
          episodes: selectedEpisodes,
          settings: settings,
        }),
      });

      if (response.ok) {
        onNotification('Conversation saved', 'success');
      }
    } catch (error) {
      console.error('Error saving conversation:', error);
      onNotification('Failed to save conversation', 'error');
    }
  };

  const exportConversation = () => {
    const exportData = {
      conversation: messages,
      episodes: selectedEpisodes,
      settings: settings,
      exportedAt: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${currentConversationId || 'new'}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const handleFeedback = async (messageId, rating, feedback) => {
    try {
      await fetch('/api/rag/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messageId,
          conversationId: currentConversationId,
          rating,
          feedback,
        }),
      });
      onNotification('Feedback submitted', 'success');
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
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
    <Box sx={{ maxWidth: 1400, mx: 'auto', height: 'calc(100vh - 200px)' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Main Chat
        </Typography>
        <Box>
          <IconButton onClick={() => setHistoryOpen(true)}>
            <History />
          </IconButton>
          <IconButton onClick={() => setSettingsOpen(true)}>
            <Settings />
          </IconButton>
          <Button onClick={exportConversation} startIcon={<Download />} sx={{ ml: 1 }}>
            Export
          </Button>
        </Box>
      </Box>
      
      <Grid container spacing={3} sx={{ height: '100%' }}>
        {/* Episode Selection Panel */}
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Tabs
                value={tabValue}
                onChange={(e, v) => setTabValue(v)}
                variant="fullWidth"
                sx={{ mb: 2 }}
              >
                <Tab label="Episodes" />
                <Tab label="Context" />
              </Tabs>

              {tabValue === 0 && (
                <>
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
                        sx={{ mb: 2 }}
                      >
                        Clear Selection
                      </Button>

                      <Button
                        fullWidth
                        variant="contained"
                        onClick={startNewConversation}
                        disabled={selectedEpisodes.length === 0}
                      >
                        New Conversation
                      </Button>
                    </>
                  )}
                </>
              )}

              {tabValue === 1 && (
                <Box>
                  <Typography variant="h6" gutterBottom>Context Info</Typography>
                  {currentConversationId && (
                    <>
                      <Typography variant="body2" gutterBottom>
                        Conversation ID: {currentConversationId}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        Messages: {messages.length}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        Episodes: {selectedEpisodes.length}
                      </Typography>
                      <Button 
                        fullWidth 
                        variant="outlined" 
                        onClick={saveConversation}
                        sx={{ mt: 2 }}
                      >
                        Save Conversation
                      </Button>
                    </>
                  )}
                </Box>
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
                  Advanced Chat {selectedEpisodes.length > 0 && `(${selectedEpisodes.length} episodes)`}
                  {currentConversationId && (
                    <Chip 
                      label={`Conversation ${currentConversationId.slice(0, 8)}`} 
                      size="small" 
                      sx={{ ml: 1 }} 
                    />
                  )}
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
                    Start an advanced conversation
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    This chat maintains context and provides detailed responses with sources
                  </Typography>
                </Box>
              ) : (
                messages.map((msg, index) => (
                  <Box key={msg.id} sx={{ mb: 3 }}>
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
                            p: 2, 
                            bgcolor: msg.type === 'user' ? 'primary.light' : 'background.default',
                            color: msg.type === 'user' ? 'primary.contrastText' : 'text.primary',
                          }}
                        >
                          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
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

                        {msg.suggestions && msg.suggestions.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Follow-up suggestions:
                            </Typography>
                            <Box sx={{ mt: 0.5 }}>
                              {msg.suggestions.map((suggestion, index) => (
                                <Button
                                  key={index}
                                  size="small"
                                  variant="outlined"
                                  onClick={() => setMessage(suggestion)}
                                  sx={{ mr: 0.5, mb: 0.5 }}
                                >
                                  {suggestion}
                                </Button>
                              ))}
                            </Box>
                          </Box>
                        )}

                        {msg.type === 'assistant' && settings.enableFeedback && (
                          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                            <Button
                              size="small"
                              onClick={() => handleFeedback(msg.id, 1, 'helpful')}
                            >
                              👍 Helpful
                            </Button>
                            <Button
                              size="small"
                              onClick={() => handleFeedback(msg.id, -1, 'not helpful')}
                            >
                              👎 Not Helpful
                            </Button>
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
                      Processing your request...
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
                  placeholder="Ask detailed questions about the podcast content..."
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

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Chat Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Typography gutterBottom>Context Length</Typography>
            <Slider
              value={settings.contextLength}
              onChange={(e, v) => setSettings(prev => ({ ...prev, contextLength: v }))}
              min={1}
              max={10}
              marks
              valueLabelDisplay="auto"
            />
            
            <Typography gutterBottom sx={{ mt: 2 }}>Temperature</Typography>
            <Slider
              value={settings.temperature}
              onChange={(e, v) => setSettings(prev => ({ ...prev, temperature: v }))}
              min={0}
              max={1}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
            
            <TextField
              fullWidth
              type="number"
              label="Max Tokens"
              value={settings.maxTokens}
              onChange={(e) => setSettings(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
              sx={{ mt: 2 }}
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={settings.includeSourceContext}
                  onChange={(e) => setSettings(prev => ({ ...prev, includeSourceContext: e.target.checked }))}
                />
              }
              label="Include Source Context"
              sx={{ mt: 2, display: 'block' }}
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={settings.enableFeedback}
                  onChange={(e) => setSettings(prev => ({ ...prev, enableFeedback: e.target.checked }))}
                />
              }
              label="Enable Feedback"
              sx={{ display: 'block' }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button onClick={() => setSettingsOpen(false)} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Conversation History Dialog */}
      <Dialog open={historyOpen} onClose={() => setHistoryOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Conversation History</DialogTitle>
        <DialogContent>
          <List>
            {conversations.map((conv) => (
              <ListItem 
                key={conv.id} 
                button 
                onClick={() => {
                  loadConversation(conv.id);
                  setHistoryOpen(false);
                }}
              >
                <ListItemText
                  primary={conv.title}
                  secondary={`${conv.messageCount} messages • ${new Date(conv.updatedAt).toLocaleString()}`}
                />
              </ListItem>
            ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MainChat;
