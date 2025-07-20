import React, { useState, useEffect } from 'react';
import {
  Alert,
  Snackbar,
  CircularProgress,
  Box,
  Typography,
  Button,
  LinearProgress,
} from '@mui/material';
import { 
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useHealthCheck } from '../hooks/useApi';

const ConnectionStatus = () => {
  const { isHealthy, lastCheck } = useHealthCheck();
  const [showStatus, setShowStatus] = useState(!isHealthy);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setShowStatus(!isHealthy);
  }, [isHealthy]);

  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
    window.location.reload(); // Simple retry by reloading
  };

  const handleClose = () => {
    setShowStatus(false);
  };

  if (!showStatus) return null;

  return (
    <>
      {/* Persistent banner for connection issues */}
      {!isHealthy && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 9999,
            bgcolor: 'error.main',
            color: 'white',
            p: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
          }}
        >
          <ErrorIcon />
          <Typography variant="body2">
            Backend server is not responding. Please start the backend server.
          </Typography>
          <Button
            color="inherit"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleRetry}
            sx={{ ml: 2 }}
          >
            Retry
          </Button>
        </Box>
      )}

      {/* Loading overlay when connecting */}
      {retryCount > 0 && !isHealthy && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            bgcolor: 'rgba(0, 0, 0, 0.7)',
            zIndex: 10000,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
          }}
        >
          <CircularProgress color="inherit" size={60} sx={{ mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Connecting to Backend Server...
          </Typography>
          <Typography variant="body2" sx={{ mb: 2, textAlign: 'center', maxWidth: 400 }}>
            Please make sure the backend server is running on port 8000.
            <br />
            Run: <code>python backend/main.py</code>
          </Typography>
          <LinearProgress sx={{ width: 300, mt: 2 }} />
        </Box>
      )}

      {/* Success notification */}
      <Snackbar
        open={isHealthy && retryCount > 0}
        autoHideDuration={3000}
        onClose={() => setRetryCount(0)}
      >
        <Alert
          onClose={() => setRetryCount(0)}
          severity="success"
          sx={{ width: '100%' }}
          icon={<CheckIcon />}
        >
          Backend server connected successfully!
        </Alert>
      </Snackbar>
    </>
  );
};

export default ConnectionStatus;
