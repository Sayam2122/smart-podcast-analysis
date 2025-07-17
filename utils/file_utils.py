"""
File utilities for the podcast analysis pipeline.
Handles file operations, caching, and data persistence.
"""

import json
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import soundfile as sf
from datetime import datetime
import os

from utils.logger import get_logger

logger = get_logger(__name__)


class FileUtils:
    """Utility class for file operations and caching"""
    
    def __init__(self, base_dir: str = "output"):
        """
        Initialize FileUtils
        
        Args:
            base_dir: Base directory for output files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.sessions_dir = self.base_dir / "sessions"
        self.cache_dir = self.base_dir / "cache"
        self.models_dir = self.base_dir / "models"
        
        for dir_path in [self.sessions_dir, self.cache_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_session_dir(self, session_id: str) -> Path:
        """
        Create a directory for a processing session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Path to the session directory
        """
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        return session_dir
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get path to session directory"""
        return self.sessions_dir / session_id
    
    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        if not self.sessions_dir.exists():
            return []
        
        return [d.name for d in self.sessions_dir.iterdir() if d.is_dir()]
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save data as JSON file
        
        Args:
            data: Data to save
            file_path: Path to save the file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.debug(f"Saved JSON to: {file_path}")
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded data or None if file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded JSON from: {file_path}")
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return None
    
    def save_pickle(self, data: Any, file_path: Union[str, Path]) -> None:
        """
        Save data as pickle file
        
        Args:
            data: Data to save
            file_path: Path to save the file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Saved pickle to: {file_path}")
    
    def load_pickle(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Load data from pickle file
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Loaded data or None if file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Pickle file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Loaded pickle from: {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading pickle from {file_path}: {e}")
            return None
    
    def save_audio(self, audio_data: np.ndarray, 
                   file_path: Union[str, Path],
                   sample_rate: int = 16000) -> None:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio numpy array
            file_path: Path to save the audio file
            sample_rate: Audio sample rate
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(file_path, audio_data, sample_rate)
        logger.debug(f"Saved audio to: {file_path}")
    
    def load_audio(self, file_path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """
        Load audio data from file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        audio_data, sample_rate = sf.read(file_path)
        logger.debug(f"Loaded audio from: {file_path}")
        
        return audio_data, sample_rate
    
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Get MD5 hash of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def get_cache_path(self, cache_key: str, extension: str = ".pkl") -> Path:
        """
        Get path for cached file
        
        Args:
            cache_key: Unique cache key
            extension: File extension
            
        Returns:
            Path to cache file
        """
        cache_file = f"{cache_key}{extension}"
        return self.cache_dir / cache_file
    
    def is_cached(self, cache_key: str, extension: str = ".pkl") -> bool:
        """
        Check if data is cached
        
        Args:
            cache_key: Unique cache key
            extension: File extension
            
        Returns:
            True if cached file exists
        """
        cache_path = self.get_cache_path(cache_key, extension)
        return cache_path.exists()
    
    def save_to_cache(self, data: Any, cache_key: str, extension: str = ".pkl") -> None:
        """
        Save data to cache
        
        Args:
            data: Data to cache
            cache_key: Unique cache key
            extension: File extension
        """
        cache_path = self.get_cache_path(cache_key, extension)
        
        if extension == ".pkl":
            self.save_pickle(data, cache_path)
        elif extension == ".json":
            self.save_json(data, cache_path)
        else:
            raise ValueError(f"Unsupported cache extension: {extension}")
    
    def load_from_cache(self, cache_key: str, extension: str = ".pkl") -> Optional[Any]:
        """
        Load data from cache
        
        Args:
            cache_key: Unique cache key
            extension: File extension
            
        Returns:
            Cached data or None if not found
        """
        cache_path = self.get_cache_path(cache_key, extension)
        
        if extension == ".pkl":
            return self.load_pickle(cache_path)
        elif extension == ".json":
            return self.load_json(cache_path)
        else:
            raise ValueError(f"Unsupported cache extension: {extension}")
    
    def clear_cache(self, cache_key: str = None) -> None:
        """
        Clear cache files
        
        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        if cache_key:
            for ext in [".pkl", ".json"]:
                cache_path = self.get_cache_path(cache_key, ext)
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"Cleared cache: {cache_path}")
        else:
            # Clear entire cache directory
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire cache directory")
    
    def get_session_files(self, session_id: str) -> Dict[str, Path]:
        """
        Get paths to all session files
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary mapping file types to paths
        """
        session_dir = self.get_session_dir(session_id)
        
        files = {
            'audio': session_dir / "audio.wav",
            'transcript': session_dir / "transcript.json",
            'diarization': session_dir / "diarization.json",
            'emotions_text': session_dir / "emotions_text.json",
            'emotions_audio': session_dir / "emotions_audio.json",
            'semantic_blocks': session_dir / "semantic_blocks.json",
            'summary': session_dir / "summary.json",
            'comprehensive_analysis': session_dir / "comprehensive_analysis_report.json"
        }
        
        return files
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Copy file from source to destination
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        
        logger.debug(f"Copied file: {src_path} -> {dst_path}")
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return file_path.stat().st_size
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.suffix,
            'hash': self.get_file_hash(file_path)
        }


# Global file utils instance
_file_utils = None

def get_file_utils(base_dir: str = "output") -> FileUtils:
    """
    Get global FileUtils instance
    
    Args:
        base_dir: Base directory for output files
        
    Returns:
        FileUtils instance
    """
    global _file_utils
    
    if _file_utils is None:
        _file_utils = FileUtils(base_dir)
    
    return _file_utils

# Convenience functions
def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data as JSON file"""
    get_file_utils().save_json(data, file_path)

def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    return get_file_utils().load_json(file_path)

def create_session_dir(session_id: str) -> Path:
    """Create session directory"""
    return get_file_utils().create_session_dir(session_id)
