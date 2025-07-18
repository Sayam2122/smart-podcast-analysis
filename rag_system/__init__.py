"""
Smart Audio RAG System
=====================

A comprehensive Retrieval-Augmented Generation system for podcast analysis
that processes transcripts and audio analysis data to create an intelligent
query-response system with conversation memory and multi-modal capabilities.

Features:
- Multi-file vector database with hierarchical indexing
- Conversation memory and context tracking
- Smart feedback and learning system
- Multi-modal query processing (speaker, emotion, temporal)
- Local LLM integration with no external dependencies
"""

__version__ = "1.0.0"
__author__ = "Smart Audio RAG Team"

from .core.vector_database import SmartVectorDB
from .core.query_processor import SmartQueryProcessor
from .core.conversation_memory import ConversationMemory
from .core.feedback_system import FeedbackSystem
from .interfaces.cli_interface import SmartRAGCLI
from .utils.data_loader import DataLoader
from .utils.logger import setup_logger

# Initialize logger
logger = setup_logger("smart_rag")

__all__ = [
    "SmartVectorDB",
    "SmartQueryProcessor", 
    "ConversationMemory",
    "FeedbackSystem",
    "SmartRAGCLI",
    "DataLoader",
    "logger"
]
