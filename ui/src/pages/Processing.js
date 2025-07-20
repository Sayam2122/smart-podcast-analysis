import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Button,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckIcon,
  RadioButtonChecked as ActiveIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Visibility as ViewIcon,
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useSessions, useProcessing } from '../hooks/useApi';
import { useProcessingStatus, useWebSocket } from '../hooks/useWebSocket';
import { 
  formatDuration, 
  formatRelativeTime, 
  getProcessingStepName, 
  getProcessingStepIcon,
  parseSessionId,
  downloadJSON,
} from '../utils/helpers';

const Processing = () => {
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionDetailsOpen, setSessionDetailsOpen] = useState(false);

  const { data: sessions, isLoading: sessionsLoading, refetch } = useSessions();
  const { stopProcessing, resumeSession, isStopping } = useProcessing();
  const { connected } = useWebSocket();
  const processingStatus = useProcessingStatus();

  const allSteps = [
    'audio_ingestion',
    'transcription', 
    'diarization',
    'segment_enrichment',
    'emotion_detection',
    'semantic_segmentation',
    'summarization',
  ];

  const getSessionStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'processing': return 'warning';
      default: return 'default';
    }
  };

  const getSessionStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckIcon />;
      case 'error': return <ErrorIcon />;
      case 'processing': return <ActiveIcon />;
      default: return <InfoIcon />;
    }
  };

  const handleViewSession = (session) => {
    setSelectedSession(session);
    setSessionDetailsOpen(true);
  };

  const handleStopProcessing = () => {
    stopProcessing();
  };

  const handleResumeSession = (sessionId) => {
    resumeSession(sessionId);
  };

  const handleDownloadSession = (session) => {
    downloadJSON(session, `${session.session_id}_details.json`);
  };

  const currentProgress = processingStatus.isProcessing 
    ? Math.round((processingStatus.completedSteps.length / allSteps.length) * 100)
    : 0;

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Processing Status
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Monitor pipeline processing and manage sessions
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
            disabled={sessionsLoading}
          >
            Refresh
          </Button>
          {processingStatus.isProcessing && (
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStopProcessing}
              disabled={isStopping}
            >
              Stop Processing
            </Button>
          )}
        </Box>
      </Box>

      {/* Connection Status */}
      <Alert 
        severity={connected ? "success" : "error"} 
        sx={{ mb: 3 }}
        icon={connected ? <CheckIcon /> : <ErrorIcon />}
      >
        WebSocket {connected ? 'Connected' : 'Disconnected'} - Real-time updates {connected ? 'active' : 'unavailable'}
      </Alert>

      {/* Current Processing Status */}
      {processingStatus.isProcessing && (
        <Card sx={{ mb: 3, borderColor: 'warning.main', borderWidth: 1, borderStyle: 'solid' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom color="warning.main">
              Processing in Progress
            </Typography>
            
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="text.secondary">
                  Current File: {processingStatus.currentFile || 'Unknown'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Session: {processingStatus.sessionId || 'Unknown'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Current Step: {getProcessingStepName(processingStatus.currentStep)}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h4" color="warning.main" align="right">
                  {currentProgress}%
                </Typography>
                <Typography variant="body2" color="text.secondary" align="right">
                  {processingStatus.completedSteps.length} of {allSteps.length} steps
                </Typography>
              </Grid>
            </Grid>

            <LinearProgress 
              variant="determinate" 
              value={currentProgress} 
              sx={{ mb: 2, height: 8, borderRadius: 4 }}
            />

            <Typography variant="subtitle2" gutterBottom>
              Processing Steps:
            </Typography>
            <List dense>
              {allSteps.map((step, index) => {
                const isCompleted = processingStatus.completedSteps.includes(step);
                const isCurrent = processingStatus.currentStep === step;
                
                return (
                  <ListItem key={step} sx={{ py: 0.5 }}>
                    <ListItemIcon>
                      <Typography variant="body1">
                        {getProcessingStepIcon(step)}
                      </Typography>
                    </ListItemIcon>
                    <ListItemText
                      primary={getProcessingStepName(step)}
                      secondary={`Step ${index + 1}`}
                      sx={{
                        opacity: isCompleted || isCurrent ? 1 : 0.6,
                        '& .MuiListItemText-primary': {
                          fontWeight: isCurrent ? 'bold' : 'normal',
                          color: isCompleted ? 'success.main' : isCurrent ? 'warning.main' : 'text.primary',
                        }
                      }}
                    />
                    {isCompleted && <CheckIcon color="success" />}
                    {isCurrent && <ActiveIcon color="warning" />}
                  </ListItem>
                );
              })}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {processingStatus.error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="subtitle2">Processing Error:</Typography>
          <Typography variant="body2">{processingStatus.error}</Typography>
        </Alert>
      )}

      {/* Sessions List */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Processing Sessions
          </Typography>
          
          {sessionsLoading ? (
            <Typography>Loading sessions...</Typography>
          ) : sessions && sessions.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Session ID</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Audio File</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Processing Time</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sessions.map((session) => {
                    const sessionInfo = parseSessionId(session.session_id);
                    
                    return (
                      <TableRow key={session.session_id}>
                        <TableCell>
                          <Typography variant="body2" fontFamily="monospace">
                            {session.session_id}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {sessionInfo.date} {sessionInfo.time}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Chip
                            icon={getSessionStatusIcon(session.status)}
                            label={session.status}
                            color={getSessionStatusColor(session.status)}
                            size="small"
                          />
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                            {session.audio_file ? session.audio_file.split('/').pop() : 'Unknown'}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          {session.duration ? formatDuration(session.duration) : 'N/A'}
                        </TableCell>
                        
                        <TableCell>
                          {session.created_at ? formatRelativeTime(session.created_at) : 'Unknown'}
                        </TableCell>
                        
                        <TableCell>
                          {session.total_processing_time 
                            ? formatDuration(session.total_processing_time)
                            : 'N/A'
                          }
                        </TableCell>
                        
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <IconButton
                              size="small"
                              onClick={() => handleViewSession(session)}
                              title="View Details"
                            >
                              <ViewIcon />
                            </IconButton>
                            
                            <IconButton
                              size="small"
                              onClick={() => handleDownloadSession(session)}
                              title="Download"
                            >
                              <DownloadIcon />
                            </IconButton>
                            
                            {session.status !== 'completed' && (
                              <IconButton
                                size="small"
                                onClick={() => handleResumeSession(session.session_id)}
                                title="Resume"
                                color="primary"
                              >
                                <PlayIcon />
                              </IconButton>
                            )}
                          </Box>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">
              No processing sessions found. Upload and process some audio files to see them here.
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Session Details Dialog */}
      <Dialog 
        open={sessionDetailsOpen} 
        onClose={() => setSessionDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Session Details: {selectedSession?.session_id}
        </DialogTitle>
        <DialogContent>
          {selectedSession && (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Status</Typography>
                  <Chip
                    icon={getSessionStatusIcon(selectedSession.status)}
                    label={selectedSession.status}
                    color={getSessionStatusColor(selectedSession.status)}
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Duration</Typography>
                  <Typography variant="body2">
                    {selectedSession.duration ? formatDuration(selectedSession.duration) : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Processing Time</Typography>
                  <Typography variant="body2">
                    {selectedSession.total_processing_time 
                      ? formatDuration(selectedSession.total_processing_time)
                      : 'N/A'
                    }
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Audio File</Typography>
                  <Typography variant="body2" noWrap>
                    {selectedSession.audio_file || 'Unknown'}
                  </Typography>
                </Grid>
              </Grid>

              {selectedSession.completed_steps && selectedSession.completed_steps.length > 0 && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">
                      Completed Steps ({selectedSession.completed_steps.length})
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {selectedSession.completed_steps.map((step, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Typography variant="body1">
                              {getProcessingStepIcon(step)}
                            </Typography>
                          </ListItemIcon>
                          <ListItemText
                            primary={getProcessingStepName(step)}
                            secondary={step}
                          />
                          <CheckIcon color="success" />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSessionDetailsOpen(false)}>
            Close
          </Button>
          {selectedSession && selectedSession.status !== 'completed' && (
            <Button
              variant="contained"
              startIcon={<PlayIcon />}
              onClick={() => {
                handleResumeSession(selectedSession.session_id);
                setSessionDetailsOpen(false);
              }}
            >
              Resume Processing
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Processing;
