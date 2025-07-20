const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs-extra');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');
const chokidar = require('chokidar');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../react-ui/build')));

// Configuration
const WATCH_FOLDER = path.resolve(__dirname, '../watchfolder');
const OUTPUT_FOLDER = path.resolve(__dirname, '../output');
const SESSIONS_FOLDER = path.join(OUTPUT_FOLDER, 'sessions');
const RAG_PIPELINE_PATH = path.resolve(__dirname, '../rag_pipeline');
const PIPELINE_PATH = path.resolve(__dirname, '../pipeline');

// Ensure directories exist
fs.ensureDirSync(WATCH_FOLDER);
fs.ensureDirSync(OUTPUT_FOLDER);
fs.ensureDirSync(SESSIONS_FOLDER);

// Storage for active sessions and processing queue
const activeSessions = new Map();
const processingQueue = [];
let isProcessing = false;

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, WATCH_FOLDER);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}_${file.originalname}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = /audio\/(mp3|wav|flac|m4a|ogg|aac)/;
    const allowedExtensions = /\.(mp3|wav|flac|m4a|ogg|aac)$/i;
    
    if (allowedTypes.test(file.mimetype) || allowedExtensions.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error('Only audio files are allowed!'), false);
    }
  },
  limits: {
    fileSize: 500 * 1024 * 1024 // 500MB limit
  }
});

// Watch folder monitoring
const watcher = chokidar.watch(WATCH_FOLDER, {
  ignored: /(^|[\/\\])\../, // ignore dotfiles
  persistent: true
});

watcher.on('add', (filePath) => {
  console.log(`New file detected: ${filePath}`);
  setTimeout(() => {
    // Add to processing queue after a short delay to ensure file is fully written
    addToProcessingQueue(filePath);
  }, 2000);
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Add file to processing queue
function addToProcessingQueue(filePath) {
  const sessionId = generateSessionId();
  const queueItem = {
    sessionId,
    filePath,
    status: 'queued',
    addedAt: new Date(),
  };
  
  processingQueue.push(queueItem);
  console.log(`Added to queue: ${filePath} with session ${sessionId}`);
  
  io.emit('file_queued', queueItem);
  
  if (!isProcessing) {
    processQueue();
  }
}

// Generate unique session ID
function generateSessionId() {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '').slice(0, 15);
  const uuid = uuidv4().slice(0, 8);
  return `${timestamp}_${uuid}`;
}

// Process queue sequentially
async function processQueue() {
  if (isProcessing || processingQueue.length === 0) {
    return;
  }
  
  isProcessing = true;
  
  while (processingQueue.length > 0) {
    const item = processingQueue.shift();
    await processAudioFile(item);
  }
  
  isProcessing = false;
}

// Process individual audio file through pipeline
async function processAudioFile(queueItem) {
  const { sessionId, filePath } = queueItem;
  
  console.log(`Starting processing: ${sessionId}`);
  
  const sessionInfo = {
    sessionId,
    audioFile: filePath,
    status: 'processing',
    startTime: new Date(),
    currentStep: 'initializing',
    progress: 0,
    steps: []
  };
  
  activeSessions.set(sessionId, sessionInfo);
  io.emit('processing_started', sessionInfo);
  
  try {
    // Run the pipeline
    await runPipeline(sessionId, filePath, sessionInfo);
    
    // After pipeline completion, integrate with RAG
    await integrateWithRAG(sessionId, sessionInfo);
    
    sessionInfo.status = 'completed';
    sessionInfo.completedAt = new Date();
    sessionInfo.progress = 100;
    
    console.log(`Processing completed: ${sessionId}`);
    io.emit('processing_completed', sessionInfo);
    
  } catch (error) {
    console.error(`Processing failed: ${sessionId}`, error);
    sessionInfo.status = 'error';
    sessionInfo.error = error.message;
    sessionInfo.completedAt = new Date();
    
    io.emit('processing_error', sessionInfo);
  } finally {
    activeSessions.delete(sessionId);
  }
}

// Run the main pipeline
function runPipeline(sessionId, filePath, sessionInfo) {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const scriptPath = path.join(__dirname, 'pipeline_wrapper.py');
    
    const args = [
      scriptPath,
      '--audio-file', filePath,
      '--session-id', sessionId,
      '--output-dir', OUTPUT_FOLDER
    ];
    
    const pythonProcess = spawn(pythonPath, args, {
      cwd: __dirname
    });
    
    let output = '';
    let errorOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
      
      // Parse progress updates
      const lines = data.toString().split('\n');
      lines.forEach(line => {
        if (line.includes('STEP:')) {
          const stepMatch = line.match(/STEP: (.+)/);
          if (stepMatch) {
            sessionInfo.currentStep = stepMatch[1];
            io.emit('processing_update', sessionInfo);
          }
        }
        
        if (line.includes('PROGRESS:')) {
          const progressMatch = line.match(/PROGRESS: (\d+)/);
          if (progressMatch) {
            sessionInfo.progress = parseInt(progressMatch[1]);
            io.emit('processing_update', sessionInfo);
          }
        }
      });
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error(`Pipeline stderr: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log(`Pipeline completed successfully for ${sessionId}`);
        resolve();
      } else {
        console.error(`Pipeline failed with code ${code} for ${sessionId}`);
        reject(new Error(`Pipeline failed with exit code ${code}: ${errorOutput}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error(`Pipeline process error: ${error}`);
      reject(error);
    });
  });
}

// Integrate processed results with RAG system
async function integrateWithRAG(sessionId, sessionInfo) {
  return new Promise((resolve, reject) => {
    sessionInfo.currentStep = 'rag_integration';
    io.emit('processing_update', sessionInfo);
    
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const scriptPath = path.join(__dirname, 'rag_integration.py');
    
    const args = [
      scriptPath,
      '--session-id', sessionId,
      '--sessions-dir', SESSIONS_FOLDER
    ];
    
    const pythonProcess = spawn(pythonPath, args, {
      cwd: __dirname
    });
    
    let output = '';
    let errorOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log(`RAG integration completed for ${sessionId}`);
        resolve();
      } else {
        console.error(`RAG integration failed with code ${code} for ${sessionId}`);
        reject(new Error(`RAG integration failed: ${errorOutput}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error(`RAG integration process error: ${error}`);
      reject(error);
    });
  });
}

// API Routes

// Upload files
app.post('/api/upload', upload.array('files'), (req, res) => {
  try {
    const files = req.files;
    const sessions = [];
    
    files.forEach(file => {
      const sessionId = generateSessionId();
      sessions.push({
        sessionId,
        filename: file.originalname,
        path: file.path,
        size: file.size
      });
      
      // Add to processing queue
      addToProcessingQueue(file.path);
    });
    
    res.json({
      success: true,
      message: `Uploaded ${files.length} files`,
      sessions
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Upload failed' });
  }
});

// Start processing
app.post('/api/process', (req, res) => {
  try {
    const { sessionId, config } = req.body;
    
    // Find session in queue or create new processing task
    const queueItem = processingQueue.find(item => item.sessionId === sessionId);
    
    if (queueItem) {
      queueItem.config = config;
      res.json({ success: true, message: 'Processing will start with configuration' });
    } else {
      res.status(404).json({ error: 'Session not found in queue' });
    }
  } catch (error) {
    console.error('Process start error:', error);
    res.status(500).json({ error: 'Failed to start processing' });
  }
});

// Resume session
app.post('/api/resume/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const sessionPath = path.join(SESSIONS_FOLDER, `session_${sessionId}`);
    
    if (!fs.existsSync(sessionPath)) {
      return res.status(404).json({ error: 'Session not found' });
    }
    
    // Load session state
    const statePath = path.join(sessionPath, 'state.json');
    if (fs.existsSync(statePath)) {
      const state = fs.readJsonSync(statePath);
      
      // Add to processing queue for resumption
      const queueItem = {
        sessionId,
        filePath: state.audio_file,
        status: 'queued',
        resume: true,
        addedAt: new Date(),
      };
      
      processingQueue.push(queueItem);
      
      if (!isProcessing) {
        processQueue();
      }
      
      res.json({ success: true, message: 'Session added to processing queue for resumption' });
    } else {
      res.status(400).json({ error: 'Session state not found' });
    }
  } catch (error) {
    console.error('Resume session error:', error);
    res.status(500).json({ error: 'Failed to resume session' });
  }
});

// Get all sessions
app.get('/api/sessions', async (req, res) => {
  try {
    const sessions = [];
    
    if (fs.existsSync(SESSIONS_FOLDER)) {
      const sessionDirs = fs.readdirSync(SESSIONS_FOLDER)
        .filter(dir => dir.startsWith('session_'))
        .map(dir => dir.replace('session_', ''));
      
      for (const sessionId of sessionDirs) {
        const sessionPath = path.join(SESSIONS_FOLDER, `session_${sessionId}`);
        const statePath = path.join(sessionPath, 'state.json');
        
        if (fs.existsSync(statePath)) {
          const state = fs.readJsonSync(statePath);
          sessions.push({
            sessionId,
            ...state,
            createdAt: state.created_at,
            updatedAt: state.last_updated,
            status: state.last_completed_step === 'pipeline_complete' ? 'completed' : 'incomplete'
          });
        }
      }
    }
    
    res.json(sessions);
  } catch (error) {
    console.error('Get sessions error:', error);
    res.status(500).json({ error: 'Failed to get sessions' });
  }
});

// Get session details
app.get('/api/sessions/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const sessionPath = path.join(SESSIONS_FOLDER, `session_${sessionId}`);
    
    if (!fs.existsSync(sessionPath)) {
      return res.status(404).json({ error: 'Session not found' });
    }
    
    const details = {};
    
    // Load all available files
    const files = ['state.json', 'config.json', 'final_report.json', 'complete_results.json'];
    
    for (const file of files) {
      const filePath = path.join(sessionPath, file);
      if (fs.existsSync(filePath)) {
        details[file.replace('.json', '')] = fs.readJsonSync(filePath);
      }
    }
    
    res.json(details);
  } catch (error) {
    console.error('Get session details error:', error);
    res.status(500).json({ error: 'Failed to get session details' });
  }
});

// RAG API Routes

// Get episodes
app.get('/api/rag/episodes', async (req, res) => {
  try {
    const episodes = [];
    
    if (fs.existsSync(SESSIONS_FOLDER)) {
      const sessionDirs = fs.readdirSync(SESSIONS_FOLDER)
        .filter(dir => dir.startsWith('session_'));
      
      for (const sessionDir of sessionDirs) {
        const sessionPath = path.join(SESSIONS_FOLDER, sessionDir);
        const finalReportPath = path.join(sessionPath, 'final_report.json');
        
        if (fs.existsSync(finalReportPath)) {
          const report = fs.readJsonSync(finalReportPath);
          episodes.push({
            id: sessionDir.replace('session_', ''),
            name: path.basename(report.session_info?.audio_file || sessionDir),
            segments: report.content_analysis?.total_segments || 0,
            duration: report.audio_info?.duration || 0,
            processed: true
          });
        }
      }
    }
    
    res.json({ episodes });
  } catch (error) {
    console.error('Get episodes error:', error);
    res.status(500).json({ error: 'Failed to get episodes' });
  }
});

// Quick chat
app.post('/api/rag/quick-chat', async (req, res) => {
  try {
    const { message, episodes } = req.body;
    
    // Call RAG quick chat script
    const response = await callRAGScript('quick_chat.py', {
      message,
      episodes,
      mode: 'quick'
    });
    
    res.json(response);
  } catch (error) {
    console.error('Quick chat error:', error);
    res.status(500).json({ error: 'Quick chat failed' });
  }
});

// Main chat
app.post('/api/rag/main-chat', async (req, res) => {
  try {
    const { message, episodes, conversationId, settings } = req.body;
    
    // Call RAG main chat script
    const response = await callRAGScript('main.py', {
      message,
      episodes,
      conversationId,
      settings,
      mode: 'main'
    });
    
    res.json(response);
  } catch (error) {
    console.error('Main chat error:', error);
    res.status(500).json({ error: 'Main chat failed' });
  }
});

// Extract quotes
app.post('/api/rag/extract-quotes', async (req, res) => {
  try {
    const { episodes, params } = req.body;
    
    // Call quote extraction script
    const response = await callRAGScript('extract_quotes.py', {
      episodes,
      params
    });
    
    res.json(response);
  } catch (error) {
    console.error('Extract quotes error:', error);
    res.status(500).json({ error: 'Quote extraction failed' });
  }
});

// Analytics
app.get('/api/analytics', async (req, res) => {
  try {
    const { period = 'all' } = req.query;
    
    // Call analytics script
    const response = await callRAGScript('analytics.py', { period });
    
    res.json(response);
  } catch (error) {
    console.error('Analytics error:', error);
    res.status(500).json({ error: 'Analytics failed' });
  }
});

// Helper function to call RAG scripts
function callRAGScript(scriptName, params) {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const scriptPath = path.join(RAG_PIPELINE_PATH, scriptName);
    
    const pythonProcess = spawn(pythonPath, [
      scriptPath,
      '--json-input', JSON.stringify(params)
    ], {
      cwd: RAG_PIPELINE_PATH
    });
    
    let output = '';
    let errorOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (parseError) {
          reject(new Error(`Failed to parse script output: ${parseError.message}`));
        }
      } else {
        reject(new Error(`Script failed with code ${code}: ${errorOutput}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../react-ui/build/index.html'));
});

// Start server
const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Watch folder: ${WATCH_FOLDER}`);
  console.log(`Output folder: ${OUTPUT_FOLDER}`);
});
