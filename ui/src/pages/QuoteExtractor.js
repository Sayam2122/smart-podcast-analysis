import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Chip,
  List,
  ListItem,
  ListItemText,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Alert,
  Paper,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  FormatQuote as QuoteIcon,
  FilterList as FilterIcon,
  Clear as ClearIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useSessions, useQuotes } from '../hooks/useApi';
import { formatDuration, downloadJSON, downloadCSV, truncateText } from '../utils/helpers';

const QuoteExtractor = () => {
  const [selectedSessions, setSelectedSessions] = useState([]);
  const [filters, setFilters] = useState({
    emotion: '',
    speaker: '',
    minDuration: '',
    maxDuration: '',
    searchText: '',
    includeTimestamps: true,
    includeContext: true,
  });
  const [extractedQuotes, setExtractedQuotes] = useState([]);
  const [selectedQuote, setSelectedQuote] = useState(null);
  const [quoteDetailsOpen, setQuoteDetailsOpen] = useState(false);

  const { data: sessions } = useSessions();
  const { extractQuotes, isExtracting } = useQuotes();

  const completedSessions = sessions?.filter(s => s.status === 'completed') || [];

  const handleSessionToggle = (sessionId) => {
    setSelectedSessions(prev => 
      prev.includes(sessionId)
        ? prev.filter(id => id !== sessionId)
        : [...prev, sessionId]
    );
  };

  const handleSelectAllSessions = () => {
    if (selectedSessions.length === completedSessions.length) {
      setSelectedSessions([]);
    } else {
      setSelectedSessions(completedSessions.map(s => s.session_id));
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleExtractQuotes = () => {
    if (selectedSessions.length === 0) {
      return;
    }

    extractQuotes({ 
      sessionIds: selectedSessions, 
      filters: filters 
    }).then((result) => {
      if (result.data) {
        setExtractedQuotes(result.data.quotes || []);
      }
    });
  };

  const handleClearFilters = () => {
    setFilters({
      emotion: '',
      speaker: '',
      minDuration: '',
      maxDuration: '',
      searchText: '',
      includeTimestamps: true,
      includeContext: true,
    });
  };

  const handleViewQuote = (quote) => {
    setSelectedQuote(quote);
    setQuoteDetailsOpen(true);
  };

  const handleDownloadQuotes = (format) => {
    if (extractedQuotes.length === 0) return;

    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `podcast_quotes_${timestamp}`;

    if (format === 'json') {
      downloadJSON(extractedQuotes, `${filename}.json`);
    } else if (format === 'csv') {
      const csvData = extractedQuotes.map(quote => ({
        text: quote.text,
        speaker: quote.speaker,
        emotion: quote.emotion,
        session_id: quote.session_id,
        start_time: quote.start_time,
        end_time: quote.end_time,
        duration: quote.duration,
        confidence: quote.confidence,
      }));
      downloadCSV(csvData, `${filename}.csv`);
    }
  };

  // Get unique speakers and emotions from sessions for filter options
  const availableSpeakers = new Set();
  const availableEmotions = new Set(['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Quote Extractor
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Extract meaningful quotes and segments from your processed podcasts
      </Typography>

      <Grid container spacing={3}>
        {/* Filters and Configuration */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Session Selection
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">
                  Select Sessions ({selectedSessions.length} selected)
                </Typography>
                <Button size="small" onClick={handleSelectAllSessions}>
                  {selectedSessions.length === completedSessions.length ? 'Deselect All' : 'Select All'}
                </Button>
              </Box>
              
              <FormGroup sx={{ maxHeight: 200, overflow: 'auto', mb: 2 }}>
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
                          {truncateText(session.session_id, 15)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatDuration(session.duration)}
                        </Typography>
                      </Box>
                    }
                  />
                ))}
              </FormGroup>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Filters
                </Typography>
                <IconButton size="small" onClick={handleClearFilters}>
                  <ClearIcon />
                </IconButton>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Search Text"
                    value={filters.searchText}
                    onChange={(e) => handleFilterChange('searchText', e.target.value)}
                    placeholder="Search in quote content..."
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Emotion</InputLabel>
                    <Select
                      value={filters.emotion}
                      label="Emotion"
                      onChange={(e) => handleFilterChange('emotion', e.target.value)}
                    >
                      <MenuItem value="">All Emotions</MenuItem>
                      {Array.from(availableEmotions).map(emotion => (
                        <MenuItem key={emotion} value={emotion}>
                          {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Speaker</InputLabel>
                    <Select
                      value={filters.speaker}
                      label="Speaker"
                      onChange={(e) => handleFilterChange('speaker', e.target.value)}
                    >
                      <MenuItem value="">All Speakers</MenuItem>
                      {Array.from(availableSpeakers).map(speaker => (
                        <MenuItem key={speaker} value={speaker}>
                          {speaker}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    type="number"
                    label="Min Duration (s)"
                    value={filters.minDuration}
                    onChange={(e) => handleFilterChange('minDuration', e.target.value)}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    type="number"
                    label="Max Duration (s)"
                    value={filters.maxDuration}
                    onChange={(e) => handleFilterChange('maxDuration', e.target.value)}
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={filters.includeTimestamps}
                        onChange={(e) => handleFilterChange('includeTimestamps', e.target.checked)}
                        size="small"
                      />
                    }
                    label="Include Timestamps"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={filters.includeContext}
                        onChange={(e) => handleFilterChange('includeContext', e.target.checked)}
                        size="small"
                      />
                    }
                    label="Include Context"
                  />
                </Grid>
              </Grid>

              <Button
                fullWidth
                variant="contained"
                startIcon={<SearchIcon />}
                onClick={handleExtractQuotes}
                disabled={selectedSessions.length === 0 || isExtracting}
                sx={{ mt: 2 }}
              >
                {isExtracting ? 'Extracting...' : 'Extract Quotes'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Extracted Quotes ({extractedQuotes.length})
                </Typography>
                
                {extractedQuotes.length > 0 && (
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleDownloadQuotes('csv')}
                    >
                      CSV
                    </Button>
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleDownloadQuotes('json')}
                    >
                      JSON
                    </Button>
                  </Box>
                )}
              </Box>

              {extractedQuotes.length === 0 ? (
                <Alert severity="info">
                  No quotes extracted yet. Select sessions and click "Extract Quotes" to get started.
                </Alert>
              ) : (
                <List>
                  {extractedQuotes.map((quote, index) => (
                    <ListItem 
                      key={index}
                      sx={{ 
                        mb: 1,
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        flexDirection: 'column',
                        alignItems: 'stretch',
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', width: '100%', mb: 1 }}>
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="body1" paragraph>
                            <QuoteIcon sx={{ mr: 1, verticalAlign: 'text-bottom' }} />
                            "{quote.text}"
                          </Typography>
                          
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            {quote.speaker && (
                              <Chip label={`Speaker: ${quote.speaker}`} size="small" />
                            )}
                            {quote.emotion && (
                              <Chip 
                                label={`Emotion: ${quote.emotion}`} 
                                size="small"
                                color="secondary"
                              />
                            )}
                            {quote.start_time !== undefined && (
                              <Chip 
                                label={`${formatDuration(quote.start_time)} - ${formatDuration(quote.end_time)}`}
                                size="small"
                                variant="outlined"
                              />
                            )}
                            {quote.confidence && (
                              <Chip 
                                label={`Confidence: ${(quote.confidence * 100).toFixed(1)}%`}
                                size="small"
                                variant="outlined"
                              />
                            )}
                          </Box>
                        </Box>
                        
                        <IconButton size="small" onClick={() => handleViewQuote(quote)}>
                          <ViewIcon />
                        </IconButton>
                      </Box>
                      
                      <Typography variant="caption" color="text.secondary">
                        Session: {truncateText(quote.session_id, 30)}
                      </Typography>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quote Details Dialog */}
      <Dialog 
        open={quoteDetailsOpen} 
        onClose={() => setQuoteDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Quote Details</DialogTitle>
        <DialogContent>
          {selectedQuote && (
            <Box>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                <Typography variant="h6" gutterBottom>
                  <QuoteIcon sx={{ mr: 1, verticalAlign: 'text-bottom' }} />
                  "{selectedQuote.text}"
                </Typography>
              </Paper>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Speaker</Typography>
                  <Typography variant="body2">{selectedQuote.speaker || 'Unknown'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Emotion</Typography>
                  <Typography variant="body2">{selectedQuote.emotion || 'Unknown'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Time Range</Typography>
                  <Typography variant="body2">
                    {selectedQuote.start_time !== undefined 
                      ? `${formatDuration(selectedQuote.start_time)} - ${formatDuration(selectedQuote.end_time)}`
                      : 'Unknown'
                    }
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Duration</Typography>
                  <Typography variant="body2">
                    {selectedQuote.duration ? formatDuration(selectedQuote.duration) : 'Unknown'}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2">Session</Typography>
                  <Typography variant="body2" fontFamily="monospace">
                    {selectedQuote.session_id}
                  </Typography>
                </Grid>
                {selectedQuote.context && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2">Context</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {selectedQuote.context}
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setQuoteDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default QuoteExtractor;
