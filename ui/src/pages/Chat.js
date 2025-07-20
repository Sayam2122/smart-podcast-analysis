import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  Avatar,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Divider,
  Alert,
  CircularProgress,
  IconButton,
  Tabs,
  Tab,
  Switch,
} from '@mui/material';
import {
  Send as SendIcon,
  Person as PersonIcon,
  SmartToy as BotIcon,
  Clear as ClearIcon,
  Settings as SettingsIcon,
  Chat as QuickChatIcon,
  Forum as MainChatIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useSessions } from '../hooks/useApi';
import { useChat, useWebSocket } from '../hooks/useWebSocket';
import { formatRelativeTime, truncateText } from '../utils/helpers';
import ReactMarkdown from 'react-markdown';

const Chat = () => {
  const [activeTab, setActiveTab] = useState(0); // 0 = Quick Chat, 1 = Main Chat
  const [message, setMessage] = useState('');
  const [selectedSessions, setSelectedSessions] = useState([]);
  const [chatMode, setChatMode] = useState('all'); // 'all' or 'selected'
  const [conversationId, setConversationId] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  
  const messagesEndRef = useRef(null);
  const { data: sessions } = useSessions();
  const { connected } = useWebSocket();
  const { messages, isLoading, error, sendMessage, clearMessages } = useChat();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  // Generate conversation ID for main chat
  useEffect(() => {
    if (activeTab === 1 && !conversationId) {
      setConversationId(`conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    }
  }, [activeTab, conversationId]);

  const handleSendMessage = () => {
    if (!message.trim() || isLoading) return;

    const sessionIds = chatMode === 'selected' ? selectedSessions : [];
    const chatType = activeTab === 0 ? 'quick' : 'main';
    
    sendMessage(message, chatType, sessionIds);
    setMessage('');
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleSessionToggle = (sessionId) => {
    setSelectedSessions(prev => 
      prev.includes(sessionId)
        ? prev.filter(id => id !== sessionId)
        : [...prev, sessionId]
    );
  };

  const handleSelectAllSessions = () => {
    if (selectedSessions.length === sessions?.length) {
      setSelectedSessions([]);
    } else {
      setSelectedSessions(sessions?.map(s => s.session_id) || []);
    }
  };

  const completedSessions = sessions?.filter(s => s.status === 'completed') || [];
  const availableSessionsCount = chatMode === 'selected' ? selectedSessions.length : completedSessions.length;

  const chatTabs = [
    {
      label: 'Quick Chat',
      icon: <QuickChatIcon />,
      description: 'Fast responses, no conversation history'
    },
    {
      label: 'Main Chat',
      icon: <MainChatIcon />,
      description: 'Contextual conversation with memory'
    }
  ];

  return (
    <Box sx={{ p: 3, height: 'calc(100vh - 120px)', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h4" gutterBottom>
        Chat with Podcasts
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Interactive AI-powered chat interface for your processed podcasts
      </Typography>

      {/* Connection Status */}
      {!connected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          WebSocket disconnected. Real-time chat may not work properly.
        </Alert>
      )}

      {/* Chat Mode Tabs */}
      <Card sx={{ mb: 2 }}>
        <CardContent sx={{ pb: 1 }}>
          <Tabs 
            value={activeTab} 
            onChange={(e, newValue) => setActiveTab(newValue)}
            sx={{ mb: 2 }}
          >
            {chatTabs.map((tab, index) => (
              <Tab
                key={index}
                icon={tab.icon}
                label={tab.label}
                iconPosition="start"
              />
            ))}
          </Tabs>
          
          <Typography variant="body2" color="text.secondary">
            {chatTabs[activeTab].description}
          </Typography>
        </CardContent>
      </Card>

      <Grid container spacing={3} sx={{ flex: 1, overflow: 'hidden' }}>
        {/* Chat Interface */}
        <Grid item xs={12} md={8} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Messages Area */}
          <Paper 
            variant="outlined" 
            sx={{ 
              flex: 1, 
              mb: 2, 
              p: 2, 
              overflow: 'auto',
              backgroundColor: 'grey.50'
            }}
          >
            {messages.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  {activeTab === 0 ? 'Quick Chat Ready' : 'Start a Conversation'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Ask questions about your podcasts. Try:
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {[
                    "What are the main topics discussed?",
                    "Who were the speakers in the podcast?",
                    "Can you summarize the key insights?",
                    "What emotions were expressed?",
                  ].map((suggestion, index) => (
                    <Chip
                      key={index}
                      label={suggestion}
                      onClick={() => setMessage(suggestion)}
                      sx={{ m: 0.5, cursor: 'pointer' }}
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>
            ) : (
              <List>
                {messages.map((msg, index) => (
                  <ListItem
                    key={msg.id || index}
                    sx={{
                      justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start',
                      mb: 1,
                    }}
                  >
                    <Box
                      sx={{
                        maxWidth: '80%',
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 1,
                        flexDirection: msg.type === 'user' ? 'row-reverse' : 'row',
                      }}
                    >
                      <Avatar
                        sx={{
                          bgcolor: msg.type === 'user' ? 'primary.main' : 'secondary.main',
                          width: 32,
                          height: 32,
                        }}
                      >
                        {msg.type === 'user' ? <PersonIcon /> : <BotIcon />}
                      </Avatar>
                      
                      <Paper
                        sx={{
                          p: 2,
                          bgcolor: msg.type === 'user' ? 'primary.main' : 'white',
                          color: msg.type === 'user' ? 'white' : 'text.primary',
                          borderRadius: 2,
                        }}
                      >
                        <Typography variant="body2">
                          {msg.type === 'assistant' ? (
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                          ) : (
                            msg.content
                          )}
                        </Typography>
                        
                        <Typography 
                          variant="caption" 
                          sx={{ 
                            display: 'block', 
                            mt: 1, 
                            opacity: 0.7,
                            textAlign: msg.type === 'user' ? 'right' : 'left'
                          }}
                        >
                          {formatRelativeTime(msg.timestamp)}
                        </Typography>
                      </Paper>
                    </Box>
                  </ListItem>
                ))}
                
                {isLoading && (
                  <ListItem sx={{ justifyContent: 'flex-start' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32 }}>
                        <BotIcon />
                      </Avatar>
                      <Paper sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <CircularProgress size={16} />
                          <Typography variant="body2">Thinking...</Typography>
                        </Box>
                      </Paper>
                    </Box>
                  </ListItem>
                )}
                
                <div ref={messagesEndRef} />
              </List>
            )}
          </Paper>

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* Message Input */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              multiline
              maxRows={3}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={`Ask about your podcasts (${availableSessionsCount} sessions available)...`}
              disabled={isLoading || !connected}
              variant="outlined"
            />
            <Button
              variant="contained"
              onClick={handleSendMessage}
              disabled={!message.trim() || isLoading || !connected}
              sx={{ minWidth: 64 }}
            >
              <SendIcon />
            </Button>
          </Box>
        </Grid>

        {/* Settings Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">Chat Settings</Typography>
                <IconButton size="small" onClick={clearMessages}>
                  <ClearIcon />
                </IconButton>
              </Box>

              {/* Session Selection */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Chat Mode</InputLabel>
                <Select
                  value={chatMode}
                  label="Chat Mode"
                  onChange={(e) => setChatMode(e.target.value)}
                >
                  <MenuItem value="all">All Completed Sessions</MenuItem>
                  <MenuItem value="selected">Selected Sessions Only</MenuItem>
                </Select>
              </FormControl>

              {chatMode === 'selected' && (
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2">
                      Select Sessions ({selectedSessions.length} selected)
                    </Typography>
                    <Button size="small" onClick={handleSelectAllSessions}>
                      {selectedSessions.length === completedSessions.length ? 'Deselect All' : 'Select All'}
                    </Button>
                  </Box>
                  
                  <FormGroup sx={{ maxHeight: 200, overflow: 'auto' }}>
                    {completedSessions.map((session) => (
                      <FormControlLabel
                        key={session.session_id}
                        control={
                          <Checkbox
                            checked={selectedSessions.includes(session.session_id)}
                            onChange={() => handleSessionToggle(session.session_id)}
                            size="small"
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body2" noWrap>
                              {truncateText(session.session_id, 20)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {session.audio_file ? session.audio_file.split('/').pop() : 'Unknown file'}
                            </Typography>
                          </Box>
                        }
                      />
                    ))}
                  </FormGroup>
                </Box>
              )}

              <Divider sx={{ my: 2 }} />

              {/* Chat Options */}
              <Typography variant="subtitle2" gutterBottom>
                Options
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    size="small"
                  />
                }
                label="Auto-scroll to new messages"
                sx={{ display: 'block', mb: 1 }}
              />

              <Divider sx={{ my: 2 }} />

              {/* Session Status */}
              <Typography variant="subtitle2" gutterBottom>
                Session Status
              </Typography>
              
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Total Sessions: {sessions?.length || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Completed: {completedSessions.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Available for Chat: {availableSessionsCount}
                </Typography>
              </Box>

              {activeTab === 1 && conversationId && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Conversation ID: {conversationId.split('_')[2]}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  size="small"
                  onClick={() => setMessage("What are the main topics discussed in the podcasts?")}
                  disabled={isLoading}
                >
                  Main Topics
                </Button>
                <Button
                  size="small"
                  onClick={() => setMessage("Can you summarize the key insights and takeaways?")}
                  disabled={isLoading}
                >
                  Key Insights
                </Button>
                <Button
                  size="small"
                  onClick={() => setMessage("What emotions were expressed by the speakers?")}
                  disabled={isLoading}
                >
                  Emotion Analysis
                </Button>
                <Button
                  size="small"
                  onClick={() => setMessage("Who were the speakers and what did they talk about?")}
                  disabled={isLoading}
                >
                  Speaker Analysis
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Chat;
