import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Tabs,
  Tab,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Mic as MicIcon,
  Chat as ChatIcon,
  FormatQuote as QuoteIcon,
  Analytics as AnalyticsIcon,
  Upload as UploadIcon,
} from '@mui/icons-material';

// Import components
import FileUpload from './components/FileUpload';
import ProcessingStatus from './components/ProcessingStatus';
import QuickChat from './components/QuickChat';
import MainChat from './components/MainChat';
import QuotesExtractor from './components/QuotesExtractor';
import Analytics from './components/Analytics';
import { PipelineProvider } from './contexts/PipelineContext';

const App = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentTab, setCurrentTab] = useState(0);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  // Map paths to tab indices
  const pathToTab = {
    '/': 0,
    '/upload': 0,
    '/processing': 1,
    '/quick-chat': 2,
    '/main-chat': 3,
    '/quotes': 4,
    '/analytics': 5,
  };

  useEffect(() => {
    const tabIndex = pathToTab[location.pathname] || 0;
    setCurrentTab(tabIndex);
  }, [location.pathname]);

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
    const paths = ['/', '/processing', '/quick-chat', '/main-chat', '/quotes', '/analytics'];
    navigate(paths[newValue]);
  };

  const showNotification = (message, severity = 'info') => {
    setNotification({ open: true, message, severity });
  };

  const hideNotification = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <PipelineProvider>
      <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
        <AppBar position="static" sx={{ backgroundColor: '#1e293b' }}>
          <Toolbar>
            <MicIcon sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🎙️ PodSmith - Advanced Podcast Analysis
            </Typography>
          </Toolbar>
        </AppBar>

        <Box sx={{ borderBottom: 1, borderColor: 'divider', backgroundColor: '#334155' }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            aria-label="podcast analysis tabs"
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              '& .MuiTab-root': {
                color: '#cbd5e1',
                '&.Mui-selected': {
                  color: '#6366f1',
                },
              },
            }}
          >
            <Tab icon={<UploadIcon />} label="Upload & Process" />
            <Tab icon={<MicIcon />} label="Processing Status" />
            <Tab icon={<ChatIcon />} label="Quick Chat" />
            <Tab icon={<ChatIcon />} label="Main Chat" />
            <Tab icon={<QuoteIcon />} label="Quotes & Extracts" />
            <Tab icon={<AnalyticsIcon />} label="Analytics" />
          </Tabs>
        </Box>

        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          <Routes>
            <Route path="/" element={<FileUpload onNotification={showNotification} />} />
            <Route path="/upload" element={<FileUpload onNotification={showNotification} />} />
            <Route path="/processing" element={<ProcessingStatus onNotification={showNotification} />} />
            <Route path="/quick-chat" element={<QuickChat onNotification={showNotification} />} />
            <Route path="/main-chat" element={<MainChat onNotification={showNotification} />} />
            <Route path="/quotes" element={<QuotesExtractor onNotification={showNotification} />} />
            <Route path="/analytics" element={<Analytics onNotification={showNotification} />} />
          </Routes>
        </Container>

        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={hideNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={hideNotification} severity={notification.severity} sx={{ width: '100%' }}>
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </PipelineProvider>
  );
};

export default App;
