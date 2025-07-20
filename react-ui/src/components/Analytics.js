import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tab,
  Tabs,
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
  Area,
  AreaChart,
} from 'recharts';
import {
  TrendingUp,
  People,
  Schedule,
  Mood,
  Topic,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

const Analytics = ({ onNotification }) => {
  const [loading, setLoading] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState('all');
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    loadAnalytics();
  }, [selectedPeriod]);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/analytics?period=${selectedPeriod}`);
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data);
      } else {
        throw new Error('Failed to load analytics');
      }
    } catch (error) {
      console.error('Error loading analytics:', error);
      onNotification('Failed to load analytics', 'error');
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#f97316'];

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading analytics...</Typography>
      </Box>
    );
  }

  if (!analytics) {
    return (
      <Alert severity="error">
        Failed to load analytics data
      </Alert>
    );
  }

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Analytics Dashboard
        </Typography>
        <FormControl sx={{ minWidth: 120 }}>
          <InputLabel>Period</InputLabel>
          <Select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            label="Period"
          >
            <MenuItem value="all">All Time</MenuItem>
            <MenuItem value="7d">Last 7 Days</MenuItem>
            <MenuItem value="30d">Last 30 Days</MenuItem>
            <MenuItem value="90d">Last 90 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <AnalyticsIcon color="primary" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h4">{analytics.summary.totalEpisodes}</Typography>
                  <Typography color="text.secondary">Total Episodes</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Schedule color="secondary" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h4">{analytics.summary.totalDuration}</Typography>
                  <Typography color="text.secondary">Total Duration</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <People color="info" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h4">{analytics.summary.totalSpeakers}</Typography>
                  <Typography color="text.secondary">Unique Speakers</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TrendingUp color="success" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h4">{analytics.summary.averageProcessingTime}</Typography>
                  <Typography color="text.secondary">Avg Process Time</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
          <Tab label="Content Analysis" />
          <Tab label="Processing Stats" />
          <Tab label="Speaker Analytics" />
          <Tab label="Emotion Analysis" />
        </Tabs>
      </Box>

      {/* Content Analysis Tab */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Episodes by Duration
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.content.episodesByDuration}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="duration" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Content Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={analytics.content.contentTypes}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {analytics.content.contentTypes.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Top Topics
              </Typography>
              <List>
                {analytics.content.topTopics.map((topic, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={topic.name}
                      secondary={`${topic.frequency} mentions • ${topic.sentiment} sentiment`}
                    />
                    <Chip 
                      label={topic.frequency} 
                      color="primary" 
                      variant="outlined" 
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Processing Stats Tab */}
      {tabValue === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Processing Time Trends
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analytics.processing.timelineData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="avgTime" stroke="#6366f1" name="Avg Time (s)" />
                  <Line type="monotone" dataKey="episodeCount" stroke="#f59e0b" name="Episodes" />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Processing Steps Performance
              </Typography>
              <List>
                {analytics.processing.stepPerformance.map((step, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={step.name}
                      secondary={`${step.avgTime}s avg • ${step.successRate}% success`}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Processing Efficiency Over Time
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={analytics.processing.efficiencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="efficiency" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Speaker Analytics Tab */}
      {tabValue === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Speaker Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={analytics.speakers.distribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="percentage"
                  >
                    {analytics.speakers.distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Speaking Time by Episode
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.speakers.timeByEpisode}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {analytics.speakers.uniqueSpeakers.map((speaker, index) => (
                    <Bar key={speaker} dataKey={speaker} fill={COLORS[index % COLORS.length]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Speaker Statistics
              </Typography>
              <Grid container spacing={2}>
                {analytics.speakers.stats.map((speaker, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6">{speaker.name}</Typography>
                        <Typography color="text.secondary">
                          Total Time: {speaker.totalTime}
                        </Typography>
                        <Typography color="text.secondary">
                          Episodes: {speaker.episodeCount}
                        </Typography>
                        <Typography color="text.secondary">
                          Avg per Episode: {speaker.avgTimePerEpisode}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Emotion Analysis Tab */}
      {tabValue === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Overall Emotion Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={analytics.emotions.distribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {analytics.emotions.distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Emotion Trends Over Time
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analytics.emotions.trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="positive" stroke="#10b981" name="Positive" />
                  <Line type="monotone" dataKey="neutral" stroke="#6b7280" name="Neutral" />
                  <Line type="monotone" dataKey="negative" stroke="#ef4444" name="Negative" />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Emotion by Speaker
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.emotions.bySpeaker}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="speaker" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="positive" fill="#10b981" name="Positive" />
                  <Bar dataKey="neutral" fill="#6b7280" name="Neutral" />
                  <Bar dataKey="negative" fill="#ef4444" name="Negative" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default Analytics;
