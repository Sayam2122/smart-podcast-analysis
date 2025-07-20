import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
  Button,
  Grid,
  Card,
  CardContent,
  IconButton,
  Collapse,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  ExpandMore,
  ExpandLess,
  CheckCircle,
  Error,
  Schedule,
  Info,
} from '@mui/icons-material';
import { usePipeline } from '../contexts/PipelineContext';

const ProcessingStatus = ({ onNotification }) => {
  const { 
    activeSessions, 
    completedSessions, 
    currentProcessing, 
    getSessions, 
    resumeSession,
    connected 
  } = usePipeline();
  
  const [expandedSessions, setExpandedSessions] = useState({});
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [selectedSession, setSelectedSession] = useState(null);
  const [allSessions, setAllSessions] = useState([]);

  useEffect(() => {
    loadSessions();
    const interval = setInterval(loadSessions, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadSessions = async () => {
    try {
      const sessions = await getSessions();
      setAllSessions(sessions);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const toggleExpanded = (sessionId) => {
    setExpandedSessions(prev => ({
      ...prev,
      [sessionId]: !prev[sessionId]
    }));
  };

  const handleResumeSession = async (sessionId) => {
    try {
      await resumeSession(sessionId);
      onNotification('Session resumed successfully', 'success');
    } catch (error) {
      onNotification('Failed to resume session', 'error');
    }
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'processing': return 'primary';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle color="success" />;
      case 'error': return <Error color="error" />;
      case 'processing': return <Schedule color="primary" />;
      case 'paused': return <Pause color="warning" />;
      default: return <Info />;
    }
  };

  const renderProcessingStep = (step, index, total) => {
    const progress = ((index + 1) / total) * 100;
    return (
      <Box key={step.name} sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2">{step.name}</Typography>
          <Typography variant="body2" color="text.secondary">
            {step.status === 'completed' ? formatDuration(step.duration) : step.status}
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={step.status === 'completed' ? 100 : step.status === 'processing' ? 50 : 0}
          color={step.status === 'completed' ? 'success' : 'primary'}
        />
      </Box>
    );
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Processing Status
        </Typography>
        <Button startIcon={<Refresh />} onClick={loadSessions}>
          Refresh
        </Button>
      </Box>

      {!connected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Not connected to processing server. Real-time updates are not available.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Current Processing */}
        {currentProcessing && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Currently Processing
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                  {currentProcessing.audioFile}
                </Typography>
                <Chip 
                  label={currentProcessing.currentStep || 'Initializing'} 
                  color="primary"
                  variant="outlined"
                />
              </Box>
              
              {currentProcessing.steps && (
                <Box sx={{ mt: 2 }}>
                  {currentProcessing.steps.map((step, index) => 
                    renderProcessingStep(step, index, currentProcessing.steps.length)
                  )}
                </Box>
              )}
              
              <LinearProgress 
                sx={{ mt: 2 }} 
                variant="determinate" 
                value={currentProcessing.progress || 0} 
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {Math.round(currentProcessing.progress || 0)}% complete
              </Typography>
            </Paper>
          </Grid>
        )}

        {/* Active Sessions */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Active Sessions ({activeSessions.length})
            </Typography>
            {activeSessions.length === 0 ? (
              <Typography color="text.secondary">No active processing sessions</Typography>
            ) : (
              <List>
                {activeSessions.map((session) => (
                  <ListItem key={session.sessionId} sx={{ flexDirection: 'column', alignItems: 'stretch' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                      <ListItemIcon>
                        {getStatusIcon(session.status)}
                      </ListItemIcon>
                      <ListItemText
                        primary={session.audioFile || session.sessionId}
                        secondary={`Started: ${new Date(session.startTime).toLocaleString()}`}
                      />
                      <Chip 
                        label={session.status} 
                        color={getStatusColor(session.status)} 
                        size="small" 
                      />
                      <IconButton onClick={() => toggleExpanded(session.sessionId)}>
                        {expandedSessions[session.sessionId] ? <ExpandLess /> : <ExpandMore />}
                      </IconButton>
                    </Box>
                    <Collapse in={expandedSessions[session.sessionId]}>
                      <Box sx={{ mt: 2, pl: 4 }}>
                        <Typography variant="body2" gutterBottom>
                          Current Step: {session.currentStep || 'Initializing'}
                        </Typography>
                        {session.progress && (
                          <LinearProgress 
                            variant="determinate" 
                            value={session.progress} 
                            sx={{ mb: 1 }} 
                          />
                        )}
                        <Button 
                          size="small" 
                          onClick={() => {
                            setSelectedSession(session);
                            setDetailsOpen(true);
                          }}
                        >
                          View Details
                        </Button>
                      </Box>
                    </Collapse>
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        {/* Completed Sessions */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Completed ({completedSessions.length})
            </Typography>
            {completedSessions.length === 0 ? (
              <Typography color="text.secondary">No completed sessions</Typography>
            ) : (
              <List>
                {completedSessions.slice(-5).map((session) => (
                  <ListItem key={session.sessionId}>
                    <ListItemIcon>
                      {getStatusIcon(session.error ? 'error' : 'completed')}
                    </ListItemIcon>
                    <ListItemText
                      primary={session.audioFile || session.sessionId}
                      secondary={
                        <>
                          <Typography variant="body2">
                            Completed: {new Date(session.completedAt).toLocaleString()}
                          </Typography>
                          {session.totalTime && (
                            <Typography variant="body2">
                              Total time: {formatDuration(session.totalTime)}
                            </Typography>
                          )}
                        </>
                      }
                    />
                    <Chip 
                      label={session.error ? 'Error' : 'Completed'} 
                      color={session.error ? 'error' : 'success'} 
                      size="small" 
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        {/* All Sessions Overview */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              All Sessions
            </Typography>
            {allSessions.length === 0 ? (
              <Typography color="text.secondary">No sessions found</Typography>
            ) : (
              <Grid container spacing={2}>
                {allSessions.map((session) => (
                  <Grid item xs={12} sm={6} md={4} key={session.sessionId}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
                          <Typography variant="subtitle1" noWrap>
                            {session.audioFile?.split('/').pop() || session.sessionId.slice(0, 12)}
                          </Typography>
                          <Chip 
                            label={session.status} 
                            color={getStatusColor(session.status)} 
                            size="small" 
                          />
                        </Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {new Date(session.createdAt).toLocaleDateString()}
                        </Typography>
                        {session.lastStep && (
                          <Typography variant="body2">
                            Last: {session.lastStep}
                          </Typography>
                        )}
                        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                          {session.status === 'paused' && (
                            <Button 
                              size="small" 
                              startIcon={<PlayArrow />}
                              onClick={() => handleResumeSession(session.sessionId)}
                            >
                              Resume
                            </Button>
                          )}
                          <Button 
                            size="small" 
                            onClick={() => {
                              setSelectedSession(session);
                              setDetailsOpen(true);
                            }}
                          >
                            Details
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Session Details Dialog */}
      <Dialog open={detailsOpen} onClose={() => setDetailsOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Session Details: {selectedSession?.sessionId}
        </DialogTitle>
        <DialogContent>
          {selectedSession && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Audio File:</Typography>
                  <Typography variant="body2">{selectedSession.audioFile || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Status:</Typography>
                  <Chip label={selectedSession.status} color={getStatusColor(selectedSession.status)} size="small" />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Created:</Typography>
                  <Typography variant="body2">{new Date(selectedSession.createdAt).toLocaleString()}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Last Updated:</Typography>
                  <Typography variant="body2">{new Date(selectedSession.updatedAt).toLocaleString()}</Typography>
                </Grid>
                {selectedSession.totalTime && (
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">Processing Time:</Typography>
                    <Typography variant="body2">{formatDuration(selectedSession.totalTime)}</Typography>
                  </Grid>
                )}
              </Grid>
              
              {selectedSession.steps && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>Processing Steps:</Typography>
                  {selectedSession.steps.map((step, index) => (
                    <Box key={step.name} sx={{ mb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">{step.name}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {step.status} {step.duration && `(${formatDuration(step.duration)})`}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
              
              {selectedSession.error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  <Typography variant="subtitle2">Error:</Typography>
                  <Typography variant="body2">{selectedSession.error}</Typography>
                </Alert>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProcessingStatus;
