import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  LinearProgress,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Divider,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  AudioFile as AudioIcon,
  Delete as DeleteIcon,
  PlayArrow as StartIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useUpload, useWatchFolder, useProcessing } from '../hooks/useApi';
import { formatFileSize, isAudioFile } from '../utils/helpers';
import toast from 'react-hot-toast';

const Upload = () => {
  const [configOpen, setConfigOpen] = useState(false);
  const [processingConfig, setProcessingConfig] = useState({
    transcription_model: 'medium',
    device: 'auto',
    num_speakers: null,
    max_speakers: 8,
  });

  const { upload, isUploading, uploadProgress } = useUpload();
  const { data: watchFolderData, refetch: refetchWatchFolder } = useWatchFolder();
  const { startProcessing, isStarting } = useProcessing();

  const watchFolderFiles = watchFolderData?.files || [];
  const audioFiles = watchFolderFiles.filter(file => isAudioFile(file.filename));
  const nonAudioFiles = watchFolderFiles.filter(file => !isAudioFile(file.filename));

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length === 0) {
      toast.error('Please select valid audio files');
      return;
    }

    const invalidFiles = acceptedFiles.filter(file => !isAudioFile(file.name));
    if (invalidFiles.length > 0) {
      toast.error(`Invalid file types: ${invalidFiles.map(f => f.name).join(', ')}`);
      return;
    }

    upload({ files: acceptedFiles });
  }, [upload]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.mp4', '.wma']
    },
    multiple: true,
    disabled: isUploading,
  });

  const handleStartProcessing = () => {
    if (audioFiles.length === 0) {
      toast.error('No audio files to process');
      return;
    }

    const filenames = audioFiles.map(file => file.filename);
    startProcessing({ 
      files: filenames, 
      config: processingConfig 
    });
  };

  const handleConfigChange = (field, value) => {
    setProcessingConfig(prev => ({
      ...prev,
      [field]: value === '' ? null : value,
    }));
  };

  const dropzoneStyle = {
    border: `2px dashed ${isDragActive ? '#1976d2' : isDragReject ? '#f44336' : '#ccc'}`,
    borderRadius: 2,
    padding: 4,
    textAlign: 'center',
    cursor: isUploading ? 'not-allowed' : 'pointer',
    backgroundColor: isDragActive ? '#f5f5f5' : 'transparent',
    transition: 'all 0.2s ease',
    opacity: isUploading ? 0.6 : 1,
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Upload & Process Audio Files
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload your podcast audio files and start the analysis pipeline
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Audio Files
              </Typography>
              
              <Box {...getRootProps()} sx={dropzoneStyle}>
                <input {...getInputProps()} />
                <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop files here' : 'Drag & drop audio files here'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  or click to browse files
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Supported formats: MP3, M4A, WAV, FLAC, OGG, AAC, MP4, WMA
                </Typography>
              </Box>

              {isUploading && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Uploading... {uploadProgress}%
                  </Typography>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Processing Configuration */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">
                  Processing Configuration
                </Typography>
                <Button
                  startIcon={<SettingsIcon />}
                  onClick={() => setConfigOpen(true)}
                  size="small"
                >
                  Advanced
                </Button>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Transcription Model</InputLabel>
                    <Select
                      value={processingConfig.transcription_model}
                      label="Transcription Model"
                      onChange={(e) => handleConfigChange('transcription_model', e.target.value)}
                    >
                      <MenuItem value="tiny">Tiny (fastest, least accurate)</MenuItem>
                      <MenuItem value="base">Base (balanced)</MenuItem>
                      <MenuItem value="small">Small (good quality)</MenuItem>
                      <MenuItem value="medium">Medium (recommended)</MenuItem>
                      <MenuItem value="large">Large (best quality, slowest)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Device</InputLabel>
                    <Select
                      value={processingConfig.device}
                      label="Device"
                      onChange={(e) => handleConfigChange('device', e.target.value)}
                    >
                      <MenuItem value="auto">Auto (recommended)</MenuItem>
                      <MenuItem value="cpu">CPU only</MenuItem>
                      <MenuItem value="cuda">GPU (CUDA)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button
                  variant="contained"
                  startIcon={<StartIcon />}
                  onClick={handleStartProcessing}
                  disabled={audioFiles.length === 0 || isStarting}
                  size="large"
                >
                  {isStarting ? 'Starting...' : `Process ${audioFiles.length} Files`}
                </Button>
                
                {audioFiles.length > 0 && (
                  <Chip 
                    label={`${audioFiles.length} file${audioFiles.length > 1 ? 's' : ''} ready`}
                    color="success"
                    variant="outlined"
                  />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Watch Folder Files */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Files in Watch Folder
              </Typography>
              
              {audioFiles.length === 0 && nonAudioFiles.length === 0 ? (
                <Alert severity="info" icon={<InfoIcon />}>
                  No files in watch folder. Upload some audio files to get started.
                </Alert>
              ) : (
                <>
                  {audioFiles.length > 0 && (
                    <>
                      <Typography variant="subtitle2" color="success.main" gutterBottom>
                        Audio Files ({audioFiles.length})
                      </Typography>
                      <List dense>
                        {audioFiles.map((file, index) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <AudioIcon color="primary" />
                            </ListItemIcon>
                            <ListItemText
                              primary={file.filename}
                              secondary={`${formatFileSize(file.size)} • Modified: ${new Date(file.modified).toLocaleString()}`}
                            />
                            <ListItemSecondaryAction>
                              <IconButton edge="end" size="small">
                                <DeleteIcon />
                              </IconButton>
                            </ListItemSecondaryAction>
                          </ListItem>
                        ))}
                      </List>
                    </>
                  )}

                  {nonAudioFiles.length > 0 && (
                    <>
                      {audioFiles.length > 0 && <Divider sx={{ my: 2 }} />}
                      <Typography variant="subtitle2" color="warning.main" gutterBottom>
                        Other Files ({nonAudioFiles.length})
                      </Typography>
                      <List dense>
                        {nonAudioFiles.map((file, index) => (
                          <ListItem key={index}>
                            <ListItemText
                              primary={file.filename}
                              secondary={`${formatFileSize(file.size)} • Not an audio file`}
                            />
                            <ListItemSecondaryAction>
                              <IconButton edge="end" size="small">
                                <DeleteIcon />
                              </IconButton>
                            </ListItemSecondaryAction>
                          </ListItem>
                        ))}
                      </List>
                    </>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* Processing Info */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Pipeline
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                The processing pipeline will analyze your audio files through the following steps:
              </Typography>
              <List dense>
                {[
                  '🎵 Audio Ingestion - Load and normalize audio',
                  '📝 Transcription - Convert speech to text',
                  '👥 Speaker Diarization - Identify different speakers',
                  '🔗 Segment Enrichment - Combine transcription with speakers',
                  '😊 Emotion Detection - Analyze emotional content',
                  '📊 Semantic Segmentation - Group content by topics',
                  '📋 Summarization - Generate detailed summaries',
                ].map((step, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemText 
                      primary={step}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Advanced Configuration Dialog */}
      <Dialog open={configOpen} onClose={() => setConfigOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Advanced Processing Configuration</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Transcription Model</InputLabel>
                <Select
                  value={processingConfig.transcription_model}
                  label="Transcription Model"
                  onChange={(e) => handleConfigChange('transcription_model', e.target.value)}
                >
                  <MenuItem value="tiny">Tiny - Fastest, least accurate</MenuItem>
                  <MenuItem value="base">Base - Balanced speed and accuracy</MenuItem>
                  <MenuItem value="small">Small - Good quality</MenuItem>
                  <MenuItem value="medium">Medium - Recommended for most use cases</MenuItem>
                  <MenuItem value="large">Large - Best quality, slowest</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Processing Device</InputLabel>
                <Select
                  value={processingConfig.device}
                  label="Processing Device"
                  onChange={(e) => handleConfigChange('device', e.target.value)}
                >
                  <MenuItem value="auto">Auto - Let system decide (recommended)</MenuItem>
                  <MenuItem value="cpu">CPU only - More compatible, slower</MenuItem>
                  <MenuItem value="cuda">GPU (CUDA) - Faster if available</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Number of Speakers"
                type="number"
                value={processingConfig.num_speakers || ''}
                onChange={(e) => handleConfigChange('num_speakers', e.target.value ? parseInt(e.target.value) : null)}
                helperText="Leave empty for auto-detection"
                inputProps={{ min: 1, max: 20 }}
              />
            </Grid>

            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Max Speakers"
                type="number"
                value={processingConfig.max_speakers}
                onChange={(e) => handleConfigChange('max_speakers', parseInt(e.target.value))}
                helperText="Maximum speakers to detect"
                inputProps={{ min: 1, max: 20 }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigOpen(false)}>Cancel</Button>
          <Button onClick={() => setConfigOpen(false)} variant="contained">
            Save Configuration
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Upload;
