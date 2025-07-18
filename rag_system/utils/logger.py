"""
Logger utility for Smart Audio RAG System

Provides comprehensive logging with session-based organization
and performance tracking capabilities.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "smart_rag", 
                log_level: str = "INFO",
                log_file: Optional[str] = None,
                session_id: Optional[str] = None) -> logging.Logger:
    """
    Set up a comprehensive logger for the Smart RAG system.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        session_id: Optional session ID for session-specific logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or session_id:
        if not log_file:
            # Create session-specific log file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            if session_id:
                log_file = log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            else:
                log_file = log_dir / f"smart_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always use DEBUG for file logging
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class PerformanceLogger:
    """
    Performance tracking logger for monitoring system performance.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
        self.counters = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timers[operation] = datetime.now()
        self.logger.debug(f"Started timer for: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.timers:
            duration = (datetime.now() - self.timers[operation]).total_seconds()
            self.logger.info(f"Operation '{operation}' completed in {duration:.2f}s")
            del self.timers[operation]
            return duration
        else:
            self.logger.warning(f"No timer found for operation: {operation}")
            return 0.0
    
    def increment_counter(self, metric: str, value: int = 1):
        """Increment a performance counter."""
        self.counters[metric] = self.counters.get(metric, 0) + value
        self.logger.debug(f"Counter '{metric}': {self.counters[metric]}")
    
    def get_counter(self, metric: str) -> int:
        """Get current counter value."""
        return self.counters.get(metric, 0)
    
    def reset_counters(self):
        """Reset all performance counters."""
        self.counters.clear()
        self.logger.debug("Performance counters reset")
    
    def log_performance_summary(self):
        """Log a summary of all performance metrics."""
        self.logger.info("=== Performance Summary ===")
        for metric, value in self.counters.items():
            self.logger.info(f"  {metric}: {value}")
        self.logger.info("===========================")


# Module-level logger instance
default_logger = setup_logger()
