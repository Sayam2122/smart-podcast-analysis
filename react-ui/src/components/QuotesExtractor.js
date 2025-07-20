import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  List,
  ListItem,
  ListItemText,
  Tab,
  Tabs,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Divider,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  FormatQuote,
  Download,
  Share,
  Copy,
  Refresh,
  Search,
  FilterList,
} from '@mui/icons-material';

const QuotesExtractor = ({ onNotification }) => {
  const [episodes, setEpisodes] = useState([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState([]);
  const [quotes, setQuotes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedQuote, setSelectedQuote] = useState(null);
  const [quoteDialogOpen, setQuoteDialogOpen] = useState(false);
  const [extractionParams, setExtractionParams] = useState({
    type: 'all', // 'quotes', 'key_insights', 'technical_terms', 'action_items'
    minLength: 10,
    maxLength: 500,
    emotion: 'all',
    speaker: 'all',
    includeContext: true,
  });

  useEffect(() => {
    loadEpisodes();
  }, []);

  const loadEpisodes = async () => {
    try {
      const response = await fetch('/api/rag/episodes');
      if (response.ok) {
        const data = await response.json();
        setEpisodes(data.episodes);
      }
    } catch (error) {
      console.error('Error loading episodes:', error);
      onNotification('Failed to load episodes', 'error');
    }
  };

  const extractQuotes = async () => {
    if (selectedEpisodes.length === 0) {
      onNotification('Please select episodes first', 'warning');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/rag/extract-quotes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          episodes: selectedEpisodes,
          params: extractionParams,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setQuotes(data.quotes);
        onNotification(`Extracted ${data.quotes.length} items`, 'success');
      } else {
        throw new Error('Failed to extract quotes');
      }
    } catch (error) {
      console.error('Error extracting quotes:', error);
      onNotification('Failed to extract quotes', 'error');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    onNotification('Copied to clipboard', 'success');
  };

  const downloadQuotes = () => {
    const content = quotes.map(quote => ({
      text: quote.text,
      speaker: quote.speaker,
      timestamp: quote.timestamp,
      episode: quote.episode,
      context: quote.context,
      type: quote.type,
    }));

    const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extracted_quotes_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadAsText = () => {
    const content = quotes.map((quote, index) => 
      `${index + 1}. "${quote.text}"\n   - ${quote.speaker} (${quote.episode})\n   - Time: ${quote.timestamp}\n`
    ).join('\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extracted_quotes_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const generateSocialMediaContent = async (quote) => {
    try {
      const response = await fetch('/api/rag/generate-social-content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          quote: quote.text,
          speaker: quote.speaker,
          episode: quote.episode,
          platforms: ['twitter', 'instagram', 'linkedin'],
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSelectedQuote({ ...quote, socialContent: data.content });
        setQuoteDialogOpen(true);
      }
    } catch (error) {
      console.error('Error generating social content:', error);
      onNotification('Failed to generate social content', 'error');
    }
  };

  const filteredQuotes = quotes.filter(quote =>
    quote.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
    quote.speaker.toLowerCase().includes(searchQuery.toLowerCase()) ||
    quote.episode.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getQuoteTypeColor = (type) => {
    switch (type) {
      case 'quote': return 'primary';
      case 'key_insight': return 'secondary';
      case 'technical_term': return 'info';
      case 'action_item': return 'warning';
      default: return 'default';
    }
  };

  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Quotes & Content Extractor
      </Typography>

      <Grid container spacing={3}>
        {/* Control Panel */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Extraction Settings
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Episodes</InputLabel>
              <Select
                multiple
                value={selectedEpisodes}
                onChange={(e) => setSelectedEpisodes(e.target.value)}
                label="Episodes"
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

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Content Type</InputLabel>
              <Select
                value={extractionParams.type}
                onChange={(e) => setExtractionParams(prev => ({ ...prev, type: e.target.value }))}
                label="Content Type"
              >
                <MenuItem value="all">All Content</MenuItem>
                <MenuItem value="quotes">Quotes Only</MenuItem>
                <MenuItem value="key_insights">Key Insights</MenuItem>
                <MenuItem value="technical_terms">Technical Terms</MenuItem>
                <MenuItem value="action_items">Action Items</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              type="number"
              label="Min Length"
              value={extractionParams.minLength}
              onChange={(e) => setExtractionParams(prev => ({ ...prev, minLength: parseInt(e.target.value) }))}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              type="number"
              label="Max Length"
              value={extractionParams.maxLength}
              onChange={(e) => setExtractionParams(prev => ({ ...prev, maxLength: parseInt(e.target.value) }))}
              sx={{ mb: 2 }}
            />

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Emotion Filter</InputLabel>
              <Select
                value={extractionParams.emotion}
                onChange={(e) => setExtractionParams(prev => ({ ...prev, emotion: e.target.value }))}
                label="Emotion Filter"
              >
                <MenuItem value="all">All Emotions</MenuItem>
                <MenuItem value="positive">Positive</MenuItem>
                <MenuItem value="neutral">Neutral</MenuItem>
                <MenuItem value="negative">Negative</MenuItem>
                <MenuItem value="joy">Joy</MenuItem>
                <MenuItem value="surprise">Surprise</MenuItem>
                <MenuItem value="anger">Anger</MenuItem>
                <MenuItem value="sadness">Sadness</MenuItem>
              </Select>
            </FormControl>

            <Button
              fullWidth
              variant="contained"
              startIcon={<FormatQuote />}
              onClick={extractQuotes}
              disabled={loading || selectedEpisodes.length === 0}
              sx={{ mb: 2 }}
            >
              {loading ? 'Extracting...' : 'Extract Content'}
            </Button>

            {quotes.length > 0 && (
              <Box>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<Download />}
                  onClick={downloadQuotes}
                  sx={{ mb: 1 }}
                >
                  Download JSON
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<Download />}
                  onClick={downloadAsText}
                >
                  Download Text
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Results Panel */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Extracted Content ({filteredQuotes.length})
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  size="small"
                  placeholder="Search content..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
                  }}
                />
              </Box>
            </Box>

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : filteredQuotes.length === 0 ? (
              <Alert severity="info">
                No content extracted yet. Select episodes and click "Extract Content" to begin.
              </Alert>
            ) : (
              <List>
                {filteredQuotes.map((quote, index) => (
                  <React.Fragment key={index}>
                    <ListItem
                      sx={{
                        flexDirection: 'column',
                        alignItems: 'stretch',
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        mb: 2,
                        p: 2,
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          <Chip 
                            label={quote.type || 'quote'} 
                            color={getQuoteTypeColor(quote.type)}
                            size="small" 
                          />
                          {quote.emotion && (
                            <Chip 
                              label={quote.emotion} 
                              variant="outlined"
                              size="small" 
                            />
                          )}
                          <Chip 
                            label={quote.speaker} 
                            variant="outlined"
                            size="small" 
                          />
                        </Box>
                        <Box>
                          <IconButton
                            size="small"
                            onClick={() => copyToClipboard(quote.text)}
                            title="Copy to clipboard"
                          >
                            <Copy />
                          </IconButton>
                          <IconButton
                            size="small"
                            onClick={() => generateSocialMediaContent(quote)}
                            title="Generate social media content"
                          >
                            <Share />
                          </IconButton>
                        </Box>
                      </Box>

                      <Typography variant="body1" sx={{ mb: 2, fontStyle: 'italic' }}>
                        "{quote.text}"
                      </Typography>

                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          {quote.episode} • {formatTimestamp(quote.timestamp)}
                        </Typography>
                        {quote.confidence && (
                          <Chip 
                            label={`${Math.round(quote.confidence * 100)}% confidence`} 
                            size="small" 
                            variant="outlined"
                          />
                        )}
                      </Box>

                      {quote.context && (
                        <Box sx={{ mt: 1, p: 1, bgcolor: 'action.hover', borderRadius: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Context: {quote.context}
                          </Typography>
                        </Box>
                      )}
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Quote Detail Dialog */}
      <Dialog 
        open={quoteDialogOpen} 
        onClose={() => setQuoteDialogOpen(false)} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>Quote Details & Social Media Content</DialogTitle>
        <DialogContent>
          {selectedQuote && (
            <Box>
              <Typography variant="h6" gutterBottom>Original Quote</Typography>
              <Paper sx={{ p: 2, mb: 3, bgcolor: 'action.hover' }}>
                <Typography variant="body1" sx={{ fontStyle: 'italic' }}>
                  "{selectedQuote.text}"
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  - {selectedQuote.speaker} ({selectedQuote.episode})
                </Typography>
              </Paper>

              {selectedQuote.socialContent && (
                <Box>
                  <Typography variant="h6" gutterBottom>Social Media Content</Typography>
                  <Tabs value={0} sx={{ mb: 2 }}>
                    <Tab label="Twitter" />
                    <Tab label="Instagram" />
                    <Tab label="LinkedIn" />
                  </Tabs>
                  
                  {selectedQuote.socialContent.twitter && (
                    <Paper sx={{ p: 2, mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>Twitter Post</Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {selectedQuote.socialContent.twitter.post}
                      </Typography>
                      <Button
                        size="small"
                        startIcon={<Copy />}
                        onClick={() => copyToClipboard(selectedQuote.socialContent.twitter.post)}
                      >
                        Copy
                      </Button>
                    </Paper>
                  )}

                  {selectedQuote.socialContent.instagram && (
                    <Paper sx={{ p: 2, mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>Instagram Caption</Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {selectedQuote.socialContent.instagram.caption}
                      </Typography>
                      <Button
                        size="small"
                        startIcon={<Copy />}
                        onClick={() => copyToClipboard(selectedQuote.socialContent.instagram.caption)}
                      >
                        Copy
                      </Button>
                    </Paper>
                  )}

                  {selectedQuote.socialContent.linkedin && (
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>LinkedIn Post</Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {selectedQuote.socialContent.linkedin.post}
                      </Typography>
                      <Button
                        size="small"
                        startIcon={<Copy />}
                        onClick={() => copyToClipboard(selectedQuote.socialContent.linkedin.post)}
                      >
                        Copy
                      </Button>
                    </Paper>
                  )}
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setQuoteDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default QuotesExtractor;
