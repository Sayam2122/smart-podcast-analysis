import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Button,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  AccessTime as TimeIcon,
} from '@mui/icons-material';
import { useSessions, useSystemMetrics, useHealthCheck } from '../hooks/useApi';
import { useProcessingStatus, useWebSocket } from '../hooks/useWebSocket';
import { 
  formatDuration, 
  formatRelativeTime, 
  formatFileSize,
  getProcessingStepName,
  getProcessingStepIcon,
} from '../utils/helpers';

const Dashboard = () => {
  const { data: sessions, isLoading: sessionsLoading } = useSessions();
  const { data: systemMetrics, isLoading: metricsLoading } = useSystemMetrics();
  const { isHealthy, lastCheck } = useHealthCheck();
  const { connected } = useWebSocket();
  const processingStatus = useProcessingStatus();

  const recentSessions = sessions?.slice(0, 5) || [];
  const totalSessions = sessions?.length || 0;
  const completedSessions = sessions?.filter(s => s.status === 'completed').length || 0;
  const totalDuration = sessions?.reduce((acc, s) => acc + (s.duration || 0), 0) || 0;

  const systemCards = [
    {
      title: 'System Health',
      value: isHealthy ? 'Healthy' : 'Issues Detected',
      icon: isHealthy ? <CheckIcon color="success" /> : <ErrorIcon color="error" />,
      color: isHealthy ? 'success' : 'error',
      subtitle: lastCheck ? `Last check: ${formatRelativeTime(lastCheck)}` : 'Checking...',
    },
    {
      title: 'Connection Status',
      value: connected ? 'Connected' : 'Disconnected',
      icon: connected ? <CheckIcon color="success" /> : <ErrorIcon color="error" />,
      color: connected ? 'success' : 'error',
      subtitle: 'WebSocket connection',
    },
    {
      title: 'Total Sessions',
      value: totalSessions,
      icon: <StorageIcon color="primary" />,
      color: 'primary',
      subtitle: `${completedSessions} completed`,
    },
    {
      title: 'Total Duration',
      value: formatDuration(totalDuration),
      icon: <TimeIcon color="info" />,
      color: 'info',
      subtitle: 'Processed audio time',
    },
  ];

  if (systemMetrics && !metricsLoading) {
    systemCards.push(
      {
        title: 'CPU Usage',
        value: `${systemMetrics.cpu_percent?.toFixed(1)}%`,
        icon: <SpeedIcon color="warning" />,
        color: 'warning',
        subtitle: `${systemMetrics.cpu_count} cores`,
      },
      {
        title: 'Memory Usage',
        value: `${((systemMetrics.memory_used / systemMetrics.memory_total) * 100).toFixed(1)}%`,
        icon: <StorageIcon color="secondary" />,
        color: 'secondary',
        subtitle: `${formatFileSize(systemMetrics.memory_used)} / ${formatFileSize(systemMetrics.memory_total)}`,
      }
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Welcome to SmartAudioAnalyzer - Monitor your podcast analysis system
      </Typography>

      {/* System Status Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {systemCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  {card.icon}
                  <Typography variant="h6" sx={{ ml: 1, flexGrow: 1 }}>
                    {card.title}
                  </Typography>
                </Box>
                <Typography variant="h4" color={`${card.color}.main`} gutterBottom>
                  {metricsLoading && (card.title.includes('CPU') || card.title.includes('Memory')) ? (
                    <Skeleton width={60} />
                  ) : (
                    card.value
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {card.subtitle}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Processing Status */}
      {processingStatus.isProcessing && (
        <Alert 
          severity="info" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" startIcon={<StopIcon />}>
              Stop
            </Button>
          }
        >
          <Typography variant="subtitle2" gutterBottom>
            Processing in progress: {processingStatus.currentFile || 'Unknown file'}
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            Current step: {getProcessingStepName(processingStatus.currentStep)}
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={processingStatus.progress} 
            sx={{ mt: 1 }}
          />
          <Typography variant="caption" color="text.secondary">
            {processingStatus.progress}% complete
          </Typography>
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Recent Sessions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Sessions
              </Typography>
              {sessionsLoading ? (
                <Box>
                  {[...Array(3)].map((_, i) => (
                    <Box key={i} sx={{ mb: 2 }}>
                      <Skeleton variant="text" width="60%" />
                      <Skeleton variant="text" width="40%" />
                    </Box>
                  ))}
                </Box>
              ) : recentSessions.length > 0 ? (
                <List dense>
                  {recentSessions.map((session) => (
                    <ListItem key={session.session_id} sx={{ px: 0 }}>
                      <ListItemIcon>
                        {session.status === 'completed' ? (
                          <CheckIcon color="success" />
                        ) : session.status === 'error' ? (
                          <ErrorIcon color="error" />
                        ) : (
                          <InfoIcon color="info" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={session.session_id}
                        secondary={
                          <Box>
                            <Typography variant="caption" display="block">
                              {formatRelativeTime(session.created_at)}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                              <Chip 
                                label={session.status} 
                                size="small" 
                                color={
                                  session.status === 'completed' ? 'success' :
                                  session.status === 'error' ? 'error' : 'default'
                                }
                              />
                              {session.duration && (
                                <Chip 
                                  label={formatDuration(session.duration)} 
                                  size="small" 
                                  variant="outlined"
                                />
                              )}
                            </Box>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No sessions found. Upload and process your first audio file to get started.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Processing Steps */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Pipeline
              </Typography>
              <List dense>
                {[
                  'audio_ingestion',
                  'transcription',
                  'diarization',
                  'segment_enrichment',
                  'emotion_detection',
                  'semantic_segmentation',
                  'summarization',
                ].map((step, index) => {
                  const isCompleted = processingStatus.completedSteps.includes(step);
                  const isCurrent = processingStatus.currentStep === step;
                  
                  return (
                    <ListItem key={step} sx={{ px: 0 }}>
                      <ListItemIcon>
                        <Typography variant="h6">
                          {getProcessingStepIcon(step)}
                        </Typography>
                      </ListItemIcon>
                      <ListItemText
                        primary={getProcessingStepName(step)}
                        secondary={`Step ${index + 1} of 7`}
                        sx={{
                          opacity: isCompleted || isCurrent ? 1 : 0.6,
                          fontWeight: isCurrent ? 'bold' : 'normal',
                        }}
                      />
                      {isCompleted && <CheckIcon color="success" />}
                      {isCurrent && <LinearProgress sx={{ width: 20, ml: 1 }} />}
                    </ListItem>
                  );
                })}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Box sx={{ mt: 4, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          startIcon={<PlayIcon />}
          href="/upload"
          size="large"
        >
          Start New Analysis
        </Button>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={() => window.location.reload()}
        >
          Refresh Status
        </Button>
      </Box>
    </Box>
  );
};

export default Dashboard;
