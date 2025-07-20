import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Switch,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Divider,
  Alert,
  Snackbar,
  Paper,
  Slider,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
} from '@mui/material';
import {
  Save as SaveIcon,
  RestoreFromTrash as ResetIcon,
  Folder as FolderIcon,
  CloudUpload as CloudIcon,
  Security as SecurityIcon,
  Tune as TuneIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { useSettings } from '../hooks/useApi';

const Settings = () => {
  const { data: settings, updateSettings, resetSettings, isLoading } = useSettings();
  const [localSettings, setLocalSettings] = useState(settings || {});
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [newApiKey, setNewApiKey] = useState('');

  React.useEffect(() => {
    if (settings) {
      setLocalSettings(settings);
    }
  }, [settings]);

  const handleSettingChange = (category, key, value) => {
    setLocalSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value,
      },
    }));
  };

  const handleSaveSettings = async () => {
    try {
      await updateSettings(localSettings);
      setSnackbarMessage('Settings saved successfully!');
      setSnackbarOpen(true);
    } catch (error) {
      setSnackbarMessage('Failed to save settings');
      setSnackbarOpen(true);
    }
  };

  const handleResetSettings = async () => {
    try {
      await resetSettings();
      setSnackbarMessage('Settings reset to defaults');
      setSnackbarOpen(true);
    } catch (error) {
      setSnackbarMessage('Failed to reset settings');
      setSnackbarOpen(true);
    }
  };

  const handleAddApiKey = () => {
    if (newApiKey.trim()) {
      const newKeys = [...(localSettings.security?.api_keys || []), newApiKey.trim()];
      handleSettingChange('security', 'api_keys', newKeys);
      setNewApiKey('');
    }
  };

  const handleRemoveApiKey = (index) => {
    const newKeys = (localSettings.security?.api_keys || []).filter((_, i) => i !== index);
    handleSettingChange('security', 'api_keys', newKeys);
  };

  if (isLoading) {
    return <Typography>Loading settings...</Typography>;
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Configure your podcast analysis system
      </Typography>

      <Grid container spacing={3}>
        {/* Pipeline Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TuneIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Pipeline Configuration</Typography>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Audio Quality</InputLabel>
                    <Select
                      value={localSettings.pipeline?.audio_quality || 'medium'}
                      onChange={(e) => handleSettingChange('pipeline', 'audio_quality', e.target.value)}
                    >
                      <MenuItem value="low">Low (Faster)</MenuItem>
                      <MenuItem value="medium">Medium</MenuItem>
                      <MenuItem value="high">High (Better Quality)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Max Concurrent Processes: {localSettings.pipeline?.max_concurrent || 2}
                  </Typography>
                  <Slider
                    value={localSettings.pipeline?.max_concurrent || 2}
                    onChange={(e, value) => handleSettingChange('pipeline', 'max_concurrent', value)}
                    min={1}
                    max={8}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.pipeline?.enable_diarization || true}
                        onChange={(e) => handleSettingChange('pipeline', 'enable_diarization', e.target.checked)}
                      />
                    }
                    label="Enable Speaker Diarization"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.pipeline?.enable_emotion_detection || true}
                        onChange={(e) => handleSettingChange('pipeline', 'enable_emotion_detection', e.target.checked)}
                      />
                    }
                    label="Enable Emotion Detection"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.pipeline?.auto_cleanup || false}
                        onChange={(e) => handleSettingChange('pipeline', 'auto_cleanup', e.target.checked)}
                      />
                    }
                    label="Auto Cleanup Temporary Files"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Temporary Directory"
                    value={localSettings.pipeline?.temp_dir || ''}
                    onChange={(e) => handleSettingChange('pipeline', 'temp_dir', e.target.value)}
                    placeholder="/tmp/podcast_processing"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* RAG Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CloudIcon color="secondary" sx={{ mr: 1 }} />
                <Typography variant="h6">RAG Configuration</Typography>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Vector Store Backend</InputLabel>
                    <Select
                      value={localSettings.rag?.vector_store || 'chroma'}
                      onChange={(e) => handleSettingChange('rag', 'vector_store', e.target.value)}
                    >
                      <MenuItem value="chroma">ChromaDB</MenuItem>
                      <MenuItem value="faiss">FAISS</MenuItem>
                      <MenuItem value="pinecone">Pinecone</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Embedding Model</InputLabel>
                    <Select
                      value={localSettings.rag?.embedding_model || 'sentence-transformers/all-MiniLM-L6-v2'}
                      onChange={(e) => handleSettingChange('rag', 'embedding_model', e.target.value)}
                    >
                      <MenuItem value="sentence-transformers/all-MiniLM-L6-v2">MiniLM-L6-v2 (Fast)</MenuItem>
                      <MenuItem value="sentence-transformers/all-mpnet-base-v2">MPNet-base-v2 (Better)</MenuItem>
                      <MenuItem value="openai/text-embedding-ada-002">OpenAI Ada-002</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Chunk Size: {localSettings.rag?.chunk_size || 512}
                  </Typography>
                  <Slider
                    value={localSettings.rag?.chunk_size || 512}
                    onChange={(e, value) => handleSettingChange('rag', 'chunk_size', value)}
                    min={128}
                    max={2048}
                    step={128}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Chunk Overlap: {localSettings.rag?.chunk_overlap || 50}
                  </Typography>
                  <Slider
                    value={localSettings.rag?.chunk_overlap || 50}
                    onChange={(e, value) => handleSettingChange('rag', 'chunk_overlap', value)}
                    min={0}
                    max={200}
                    step={10}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Top K Results: {localSettings.rag?.top_k || 5}
                  </Typography>
                  <Slider
                    value={localSettings.rag?.top_k || 5}
                    onChange={(e, value) => handleSettingChange('rag', 'top_k', value)}
                    min={1}
                    max={20}
                    step={1}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Vector Store Path"
                    value={localSettings.rag?.vector_store_path || ''}
                    onChange={(e) => handleSettingChange('rag', 'vector_store_path', e.target.value)}
                    placeholder="./output/chroma_db"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Security Settings */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SecurityIcon color="error" sx={{ mr: 1 }} />
                <Typography variant="h6">Security</Typography>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.security?.require_authentication || false}
                        onChange={(e) => handleSettingChange('security', 'require_authentication', e.target.checked)}
                      />
                    }
                    label="Require Authentication"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.security?.enable_rate_limiting || true}
                        onChange={(e) => handleSettingChange('security', 'enable_rate_limiting', e.target.checked)}
                      />
                    }
                    label="Enable Rate Limiting"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    API Keys
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                    <TextField
                      fullWidth
                      size="small"
                      label="New API Key"
                      value={newApiKey}
                      onChange={(e) => setNewApiKey(e.target.value)}
                      type="password"
                    />
                    <Button
                      variant="outlined"
                      startIcon={<AddIcon />}
                      onClick={handleAddApiKey}
                      disabled={!newApiKey.trim()}
                    >
                      Add
                    </Button>
                  </Box>
                  
                  <List>
                    {(localSettings.security?.api_keys || []).map((key, index) => (
                      <ListItem key={index} sx={{ px: 0 }}>
                        <ListItemText
                          primary={`API Key ${index + 1}`}
                          secondary={`****${key.slice(-4)}`}
                        />
                        <ListItemSecondaryAction>
                          <IconButton 
                            edge="end" 
                            onClick={() => handleRemoveApiKey(index)}
                            color="error"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Storage Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <FolderIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Storage</Typography>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Output Directory"
                    value={localSettings.storage?.output_dir || ''}
                    onChange={(e) => handleSettingChange('storage', 'output_dir', e.target.value)}
                    placeholder="./output"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Session Logs Directory"
                    value={localSettings.storage?.logs_dir || ''}
                    onChange={(e) => handleSettingChange('storage', 'logs_dir', e.target.value)}
                    placeholder="./logs"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Max Storage Size (GB): {localSettings.storage?.max_size_gb || 10}
                  </Typography>
                  <Slider
                    value={localSettings.storage?.max_size_gb || 10}
                    onChange={(e, value) => handleSettingChange('storage', 'max_size_gb', value)}
                    min={1}
                    max={100}
                    step={1}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.storage?.auto_backup || false}
                        onChange={(e) => handleSettingChange('storage', 'auto_backup', e.target.checked)}
                      />
                    }
                    label="Auto Backup Sessions"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.performance?.enable_caching || true}
                        onChange={(e) => handleSettingChange('performance', 'enable_caching', e.target.checked)}
                      />
                    }
                    label="Enable Result Caching"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localSettings.performance?.enable_gpu || false}
                        onChange={(e) => handleSettingChange('performance', 'enable_gpu', e.target.checked)}
                      />
                    }
                    label="Use GPU Acceleration (if available)"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Memory Limit (GB): {localSettings.performance?.memory_limit_gb || 8}
                  </Typography>
                  <Slider
                    value={localSettings.performance?.memory_limit_gb || 8}
                    onChange={(e, value) => handleSettingChange('performance', 'memory_limit_gb', value)}
                    min={2}
                    max={64}
                    step={2}
                    valueLabelDisplay="auto"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                startIcon={<ResetIcon />}
                onClick={handleResetSettings}
                color="error"
              >
                Reset to Defaults
              </Button>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveSettings}
              >
                Save Settings
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
      />
    </Box>
  );
};

export default Settings;
