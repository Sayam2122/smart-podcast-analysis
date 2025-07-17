"""
Centralized logging utility for the podcast analysis pipeline.
Provides structured logging with different levels and output formats.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import json
from datetime import datetime


class PipelineLogger:
    """Centralized logger for the entire pipeline"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 session_id: Optional[str] = None):
        """
        Initialize the pipeline logger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging
            session_id: Session ID for contextual logging
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Remove default logger
        logger.remove()
        
        # Add console handler with custom format
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<blue>Session: {extra[session_id]}</blue> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | Session: {extra[session_id]} | {message}",
                rotation="100 MB",
                retention="30 days",
                compression="zip"
            )
        
        # Bind session_id to all log messages
        self.logger = logger.bind(session_id=self.session_id)
    
    def get_logger(self, module_name: str = None):
        """
        Get a logger instance for a specific module
        
        Args:
            module_name: Name of the module requesting the logger
            
        Returns:
            Configured logger instance
        """
        if module_name:
            return self.logger.bind(module=module_name)
        return self.logger
    
    def log_processing_start(self, module: str, file_path: str = None, **kwargs):
        """Log the start of a processing step"""
        extra_info = f" | File: {file_path}" if file_path else ""
        params = f" | Params: {json.dumps(kwargs)}" if kwargs else ""
        self.logger.info(f"🚀 Starting {module} processing{extra_info}{params}")
    
    def log_processing_complete(self, module: str, duration: float, output_path: str = None, **results):
        """Log the completion of a processing step"""
        output_info = f" | Output: {output_path}" if output_path else ""
        results_info = f" | Results: {json.dumps(results)}" if results else ""
        self.logger.info(f"✅ Completed {module} processing in {duration:.2f}s{output_info}{results_info}")
    
    def log_error(self, module: str, error: Exception, context: dict = None):
        """Log an error with context"""
        context_info = f" | Context: {json.dumps(context)}" if context else ""
        self.logger.error(f"❌ Error in {module}: {str(error)}{context_info}")
    
    def log_warning(self, module: str, message: str, context: dict = None):
        """Log a warning with context"""
        context_info = f" | Context: {json.dumps(context)}" if context else ""
        self.logger.warning(f"⚠️  Warning in {module}: {message}{context_info}")
    
    def log_progress(self, module: str, current: int, total: int, item_name: str = "items"):
        """Log progress information"""
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"📊 {module} progress: {current}/{total} {item_name} ({percentage:.1f}%)")
    
    def log_memory_usage(self, module: str, memory_mb: float):
        """Log memory usage information"""
        self.logger.debug(f"💾 {module} memory usage: {memory_mb:.1f} MB")
    
    def log_model_load(self, model_name: str, model_size: str = None, load_time: float = None):
        """Log model loading information"""
        size_info = f" | Size: {model_size}" if model_size else ""
        time_info = f" | Load time: {load_time:.2f}s" if load_time else ""
        self.logger.info(f"🧠 Loaded model: {model_name}{size_info}{time_info}")
    
    def log_session_start(self, audio_file: str, config: dict = None):
        """Log the start of a new session"""
        config_info = f" | Config: {json.dumps(config)}" if config else ""
        self.logger.info(f"🎬 Starting new session | Audio: {audio_file}{config_info}")
    
    def log_session_complete(self, total_duration: float, output_dir: str):
        """Log the completion of a session"""
        self.logger.info(f"🎉 Session completed in {total_duration:.2f}s | Output: {output_dir}")


# Global logger instance
_global_logger = None

def get_logger(module_name: str = None, 
               log_level: str = "INFO", 
               log_file: str = None,
               session_id: str = None) -> logger:
    """
    Get a logger instance. Creates global logger if not exists.
    
    Args:
        module_name: Name of the module requesting the logger
        log_level: Logging level
        log_file: Optional log file path
        session_id: Session ID for contextual logging
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PipelineLogger(
            log_level=log_level,
            log_file=log_file,
            session_id=session_id
        )
    
    return _global_logger.get_logger(module_name)

def setup_session_logging(session_id: str, log_dir: str = "logs"):
    """
    Setup logging for a specific session
    
    Args:
        session_id: Unique session identifier
        log_dir: Directory to store log files
    """
    global _global_logger
    
    log_file = Path(log_dir) / f"{session_id}.log"
    _global_logger = PipelineLogger(
        log_level="INFO",
        log_file=str(log_file),
        session_id=session_id
    )
    
    return _global_logger.get_logger()

# Convenience functions
def log_info(message: str, module: str = "PIPELINE"):
    """Log info message"""
    get_logger(module).info(message)

def log_error(message: str, module: str = "PIPELINE"):
    """Log error message"""
    get_logger(module).error(message)

def log_warning(message: str, module: str = "PIPELINE"):
    """Log warning message"""
    get_logger(module).warning(message)

def log_debug(message: str, module: str = "PIPELINE"):
    """Log debug message"""
    get_logger(module).debug(message)
