import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Paper,
  Typography,
  Box,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  CloudUpload,
  AudioFile,
  Delete,
  PlayArrow,
  ExpandMore,
  Settings,
} from '@mui/icons-material';
import { usePipeline } from '../contexts/PipelineContext';

const FileUpload = ({ onNotification }) => {
  const { uploadFiles, startProcessing, getSessions, connected } = usePipeline();
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [configOpen, setConfigOpen] = useState(false);
  const [config, setConfig] = useState({
    transcription: {
      model_size: 'medium',
      backend: 'faster-whisper',
      device: 'auto',
    },
    diarization: {
      num_speakers: null,
      min_speakers: 1,
      max_speakers: 8,
    },
    emotion_detection: {
      combine_modes: true,
    },
    semantic_segmentation: {
      min_block_size: 3,
      similarity_threshold: 0.3,
    },
    summarization: {
      model_name: 'mistral:7b',
      max_tokens: 300,
      temperature: 0.3,
    },
  });

  const onDrop = useCallback((acceptedFiles) => {
    const audioFiles = acceptedFiles.filter(file => 
      file.type.startsWith('audio/') || 
      file.name.toLowerCase().match(/\.(mp3|wav|flac|m4a|ogg|aac)$/)
    );
    
    if (audioFiles.length !== acceptedFiles.length) {
      onNotification('Some files were rejected. Only audio files are accepted.', 'warning');
    }
    
    setFiles(prev => [...prev, ...audioFiles.map(file => ({
      file,
      id: Date.now() + Math.random(),
      status: 'ready',
    }))]);
  }, [onNotification]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'],
    },
    multiple: true,
  });

  const removeFile = (id) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const handleUploadAndProcess = async () => {
    if (files.length === 0) {
      onNotification('Please select audio files first', 'warning');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      // Upload files
      const fileList = files.map(f => f.file);
      const uploadResult = await uploadFiles(fileList);
      
      setUploadProgress(50);
      
      // Start processing for each uploaded file
      for (const sessionInfo of uploadResult.sessions) {
        await startProcessing(sessionInfo.sessionId, config);
      }
      
      setUploadProgress(100);
      onNotification(`Successfully uploaded and started processing ${files.length} files`, 'success');
      setFiles([]);
      
    } catch (error) {
      console.error('Upload/processing error:', error);
      onNotification('Failed to upload or start processing', 'error');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Upload Audio Files
      </Typography>
      
      {!connected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Not connected to processing server. Please ensure the backend is running.
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper
            {...getRootProps()}
            sx={{
              p: 4,
              textAlign: 'center',
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.400',
              backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'action.hover',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop audio files here' : 'Drag & drop audio files here'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supports MP3, WAV, FLAC, M4A, OGG, AAC formats
            </Typography>
            <Button variant="outlined" sx={{ mt: 2 }}>
              Choose Files
            </Button>
          </Paper>

          {files.length > 0 && (
            <Paper sx={{ mt: 3 }}>
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">
                  Selected Files ({files.length})
                </Typography>
                <Box>
                  <Button
                    startIcon={<Settings />}
                    onClick={() => setConfigOpen(true)}
                    sx={{ mr: 1 }}
                  >
                    Configure
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<PlayArrow />}
                    onClick={handleUploadAndProcess}
                    disabled={uploading || !connected}
                  >
                    Upload & Process
                  </Button>
                </Box>
              </Box>
              
              {uploading && (
                <LinearProgress 
                  variant="determinate" 
                  value={uploadProgress} 
                  sx={{ mx: 2, mb: 2 }} 
                />
              )}

              <List>
                {files.map((fileItem) => (
                  <ListItem key={fileItem.id}>
                    <ListItemIcon>
                      <AudioFile color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={fileItem.file.name}
                      secondary={
                        <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                          <Chip 
                            label={formatFileSize(fileItem.file.size)} 
                            size="small" 
                            variant="outlined" 
                          />
                          <Chip 
                            label={fileItem.status} 
                            size="small" 
                            color={fileItem.status === 'ready' ? 'primary' : 'default'}
                          />
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton 
                        edge="end" 
                        onClick={() => removeFile(fileItem.id)}
                        disabled={uploading}
                      >
                        <Delete />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            </Paper>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Pipeline
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="1. Audio Ingestion" secondary="Load and normalize audio" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="2. Transcription" secondary="Speech-to-text conversion" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="3. Speaker Diarization" secondary="Identify speakers" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="4. Emotion Detection" secondary="Analyze emotions" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="5. Semantic Segmentation" secondary="Topic segmentation" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="6. Summarization" secondary="Generate summaries" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="7. RAG Integration" secondary="Index for search" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Configuration Dialog */}
      <Dialog open={configOpen} onClose={() => setConfigOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Processing Configuration</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Transcription Settings</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      select
                      fullWidth
                      label="Model Size"
                      value={config.transcription.model_size}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        transcription: { ...prev.transcription, model_size: e.target.value }
                      }))}
                      SelectProps={{ native: true }}
                    >
                      <option value="tiny">Tiny (fastest)</option>
                      <option value="base">Base</option>
                      <option value="small">Small</option>
                      <option value="medium">Medium (recommended)</option>
                      <option value="large">Large (best quality)</option>
                    </TextField>
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      select
                      fullWidth
                      label="Backend"
                      value={config.transcription.backend}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        transcription: { ...prev.transcription, backend: e.target.value }
                      }))}
                      SelectProps={{ native: true }}
                    >
                      <option value="faster-whisper">Faster Whisper</option>
                      <option value="openai-whisper">OpenAI Whisper</option>
                    </TextField>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Speaker Diarization</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <TextField
                      type="number"
                      fullWidth
                      label="Min Speakers"
                      value={config.diarization.min_speakers}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        diarization: { ...prev.diarization, min_speakers: parseInt(e.target.value) }
                      }))}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      type="number"
                      fullWidth
                      label="Max Speakers"
                      value={config.diarization.max_speakers}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        diarization: { ...prev.diarization, max_speakers: parseInt(e.target.value) }
                      }))}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      type="number"
                      fullWidth
                      label="Num Speakers (optional)"
                      value={config.diarization.num_speakers || ''}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        diarization: { ...prev.diarization, num_speakers: e.target.value ? parseInt(e.target.value) : null }
                      }))}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Summarization</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Model Name"
                      value={config.summarization.model_name}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        summarization: { ...prev.summarization, model_name: e.target.value }
                      }))}
                    />
                  </Grid>
                  <Grid item xs={3}>
                    <TextField
                      type="number"
                      fullWidth
                      label="Max Tokens"
                      value={config.summarization.max_tokens}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        summarization: { ...prev.summarization, max_tokens: parseInt(e.target.value) }
                      }))}
                    />
                  </Grid>
                  <Grid item xs={3}>
                    <TextField
                      type="number"
                      fullWidth
                      label="Temperature"
                      inputProps={{ step: 0.1, min: 0, max: 1 }}
                      value={config.summarization.temperature}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        summarization: { ...prev.summarization, temperature: parseFloat(e.target.value) }
                      }))}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigOpen(false)}>Cancel</Button>
          <Button onClick={() => setConfigOpen(false)} variant="contained">Save Configuration</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FileUpload;
