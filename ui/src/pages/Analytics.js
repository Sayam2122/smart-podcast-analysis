import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Chip,
  Alert,
  Paper,
  CircularProgress,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area,
  ScatterPlot,
  Scatter,
} from 'recharts';
import {
  TrendingUp as TrendingUpIcon,
  Schedule as ScheduleIcon,
  People as PeopleIcon,
  Psychology as PsychologyIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useSessions, useAnalytics } from '../hooks/useApi';
import { formatDuration, downloadJSON } from '../utils/helpers';

const Analytics = () => {
  const [selectedSessions, setSelectedSessions] = useState([]);
  const [timeRange, setTimeRange] = useState('7d');
  const [analyticsType, setAnalyticsType] = useState('overview');

  const { data: sessions } = useSessions();
  const { data: analytics, isLoading: analyticsLoading } = useAnalytics({
    sessionIds: selectedSessions,
    timeRange,
    type: analyticsType,
  });

  const completedSessions = sessions?.filter(s => s.status === 'completed') || [];

  // Color schemes for charts
  const emotionColors = {
    joy: '#4CAF50',
    sadness: '#2196F3',
    anger: '#F44336',
    fear: '#9C27B0',
    surprise: '#FF9800',
    neutral: '#607D8B',
  };

  const processedData = useMemo(() => {
    if (!analytics) return null;

    // Process emotion distribution
    const emotionData = Object.entries(analytics.emotions || {}).map(([emotion, count]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      count,
      percentage: ((count / Object.values(analytics.emotions || {}).reduce((a, b) => a + b, 0)) * 100).toFixed(1),
    }));

    // Process speaker distribution
    const speakerData = Object.entries(analytics.speakers || {}).map(([speaker, data]) => ({
      speaker,
      talkTime: data.total_duration,
      segments: data.segment_count,
      percentage: ((data.total_duration / (analytics.totalDuration || 1)) * 100).toFixed(1),
    }));

    // Process processing time trends
    const processingTrends = analytics.processing_times || [];

    // Process session statistics
    const sessionStats = {
      totalSessions: analytics.total_sessions || 0,
      totalDuration: analytics.total_duration || 0,
      averageDuration: analytics.average_duration || 0,
      totalSpeakers: analytics.unique_speakers || 0,
    };

    return {
      emotions: emotionData,
      speakers: speakerData,
      processingTrends,
      sessionStats,
    };
  }, [analytics]);

  const handleSessionChange = (event) => {
    setSelectedSessions(event.target.value);
  };

  const handleExportData = () => {
    if (analytics) {
      const timestamp = new Date().toISOString().split('T')[0];
      downloadJSON(analytics, `podcast_analytics_${timestamp}.json`);
    }
  };

  if (analyticsLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Analytics Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Insights and analytics from your processed podcasts
      </Typography>

      {/* Controls */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Sessions</InputLabel>
            <Select
              multiple
              value={selectedSessions}
              onChange={handleSessionChange}
              renderValue={(selected) => 
                selected.length === 0 ? 'All Sessions' : `${selected.length} selected`
              }
            >
              {completedSessions.map((session) => (
                <MenuItem key={session.session_id} value={session.session_id}>
                  {session.session_id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Time Range</InputLabel>
            <Select value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
              <MenuItem value="1d">Last Day</MenuItem>
              <MenuItem value="7d">Last Week</MenuItem>
              <MenuItem value="30d">Last Month</MenuItem>
              <MenuItem value="90d">Last 3 Months</MenuItem>
              <MenuItem value="all">All Time</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Analytics Type</InputLabel>
            <Select value={analyticsType} onChange={(e) => setAnalyticsType(e.target.value)}>
              <MenuItem value="overview">Overview</MenuItem>
              <MenuItem value="emotions">Emotions</MenuItem>
              <MenuItem value="speakers">Speakers</MenuItem>
              <MenuItem value="performance">Performance</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={2}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExportData}
            disabled={!analytics}
          >
            Export
          </Button>
        </Grid>
      </Grid>

      {!processedData ? (
        <Alert severity="info">
          No analytics data available. Process some sessions to see analytics.
        </Alert>
      ) : (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <TrendingUpIcon color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Total Sessions</Typography>
                  </Box>
                  <Typography variant="h4" color="primary">
                    {processedData.sessionStats.totalSessions}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <ScheduleIcon color="secondary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Total Duration</Typography>
                  </Box>
                  <Typography variant="h4" color="secondary">
                    {formatDuration(processedData.sessionStats.totalDuration)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <PeopleIcon color="success.main" sx={{ mr: 1 }} />
                    <Typography variant="h6">Unique Speakers</Typography>
                  </Box>
                  <Typography variant="h4" color="success.main">
                    {processedData.sessionStats.totalSpeakers}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <PsychologyIcon color="warning.main" sx={{ mr: 1 }} />
                    <Typography variant="h6">Avg Duration</Typography>
                  </Box>
                  <Typography variant="h4" color="warning.main">
                    {formatDuration(processedData.sessionStats.averageDuration)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
          <Grid container spacing={3}>
            {/* Emotion Distribution */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Emotion Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={processedData.emotions}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ emotion, percentage }) => `${emotion}: ${percentage}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="count"
                      >
                        {processedData.emotions.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={emotionColors[entry.emotion.toLowerCase()] || '#8884d8'} 
                          />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Speaker Talk Time */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Speaker Talk Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={processedData.speakers}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="speaker" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value) => [formatDuration(value), 'Talk Time']}
                      />
                      <Legend />
                      <Bar dataKey="talkTime" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Processing Performance */}
            {processedData.processingTrends.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Processing Performance Over Time
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={processedData.processingTrends}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value, name) => [
                            name === 'processing_time' ? formatDuration(value) : value,
                            name.replace('_', ' ').toUpperCase()
                          ]}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="processing_time" stroke="#8884d8" />
                        <Line type="monotone" dataKey="file_size_mb" stroke="#82ca9d" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Detailed Emotion Breakdown */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Emotion Analysis
                  </Typography>
                  <Grid container spacing={2}>
                    {processedData.emotions.map((emotion) => (
                      <Grid item xs={6} sm={4} md={2} key={emotion.emotion}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h4" sx={{ color: emotionColors[emotion.emotion.toLowerCase()] }}>
                            {emotion.count}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {emotion.emotion}
                          </Typography>
                          <Chip 
                            label={`${emotion.percentage}%`} 
                            size="small" 
                            sx={{ mt: 1 }}
                          />
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
};

export default Analytics;
