"""
Audio ingestion module for the podcast analysis pipeline.
Handles multiple audio formats and converts them to standardized format.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class AudioIngestion:
    def load_and_normalize_audio(self, audio_file_path: str) -> dict:
        """
        Load audio file, resample, apply noise gate, and return normalized audio data.
        Args:
            audio_file_path: Path to audio file
        Returns:
            dict with keys: audio (np.ndarray), sample_rate (int), duration (float), file_size (int)
        """
        import soundfile as sf
        import numpy as np
        from scipy.signal import resample_poly
        
        logger.info(f"Loading audio file: {audio_file_path}")
        
        # Load audio
        audio, sample_rate = sf.read(audio_file_path)
        logger.info(f"Original audio: shape={audio.shape}, sample_rate={sample_rate}")
        
        # Ensure mono (convert stereo to mono)
        if len(audio.shape) > 1:
            if audio.shape[1] == 2:
                audio = audio.mean(axis=1)  # Average left and right channels
                logger.info("Converted stereo to mono")
            else:
                audio = audio[:, 0]  # Take first channel
                logger.info("Converted multi-channel to mono")
        
        # Resample to target sample rate if needed
        if sample_rate != self.target_sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {self.target_sample_rate}Hz")
            audio = resample_poly(audio, self.target_sample_rate, sample_rate)
            sample_rate = self.target_sample_rate
        
        # Normalize amplitude to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            logger.info(f"Normalized audio amplitude (max was {max_val:.4f})")
        
        # Apply noise gate
        if self.noise_gate_threshold > 0:
            audio[np.abs(audio) < self.noise_gate_threshold] = 0
            logger.info(f"Applied noise gate with threshold {self.noise_gate_threshold}")
        
        # Ensure float32 dtype
        audio = audio.astype(np.float32)
        
        duration = len(audio) / sample_rate
        file_size = os.path.getsize(audio_file_path)
        
        logger.info(f"Final audio: shape={audio.shape}, duration={duration:.2f}s, sample_rate={sample_rate}")
        
        return {
            'audio': audio,
            'sample_rate': sample_rate,
            'duration': duration,
            'file_size': file_size
        }
    """
    Audio ingestion and preprocessing module
    Converts various audio formats to standardized 16kHz mono PCM WAV
    """
    
    def __init__(self, target_sample_rate: int = 16000, noise_gate_threshold: float = 0.01):
        """
        Initialize audio ingestion
        
        Args:
            target_sample_rate: Target sample rate for processing
            noise_gate_threshold: Threshold for noise gate
        """
        self.target_sample_rate = target_sample_rate
        self.noise_gate_threshold = noise_gate_threshold
        self.file_utils = get_file_utils()
    
    def load_and_normalize(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and normalize to target format
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Tuple of (normalized_audio, sample_rate)
        """
        logger.info(f"Loading audio file: {audio_path}")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file extension
        ext = audio_path.suffix.lower()
        
        try:
            if ext in ['.wav', '.flac']:
                # Direct loading for supported formats
                audio_data, sample_rate = self._load_with_soundfile(audio_path)
            else:
                # Use pydub for other formats
                audio_data, sample_rate = self._load_with_pydub(audio_path)
            
            # Normalize audio
            normalized_audio = self._normalize_audio(audio_data, sample_rate)
            
            logger.info(f"Successfully loaded and normalized audio | "
                       f"Duration: {len(normalized_audio) / self.target_sr:.2f}s | "
                       f"Sample rate: {self.target_sr}Hz | "
                       f"Channels: 1 (mono)")
            
            return normalized_audio, self.target_sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise
    
    def _load_with_soundfile(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio using soundfile (for WAV, FLAC)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sample_rate = sf.read(str(audio_path))
            logger.debug(f"Loaded with soundfile | Shape: {audio_data.shape} | SR: {sample_rate}")
            return audio_data, sample_rate
        except Exception as e:
            logger.warning(f"Soundfile loading failed, falling back to pydub: {e}")
            return self._load_with_pydub(audio_path)
    
    def _load_with_pydub(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio using pydub (for MP3, M4A, etc.)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load with pydub
            audio_segment = AudioSegment.from_file(str(audio_path))
            
            # Convert to numpy array
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Handle stereo to mono conversion
            if audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
                audio_data = audio_data.mean(axis=1)
            
            # Normalize to [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
            
            sample_rate = audio_segment.frame_rate
            
            logger.debug(f"Loaded with pydub | Shape: {audio_data.shape} | SR: {sample_rate}")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Pydub loading failed: {e}")
            raise
    
    def _normalize_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize audio to target format (16kHz mono)
        
        Args:
            audio_data: Input audio data
            sample_rate: Input sample rate
            
        Returns:
            Normalized audio data
        """
        # Ensure mono
        if len(audio_data.shape) > 1:
            if audio_data.shape[1] == 2:
                audio_data = audio_data.mean(axis=1)
            else:
                audio_data = audio_data[:, 0]
        
        # Resample if necessary
        if sample_rate != self.target_sr:
            audio_data = self._resample_audio(audio_data, sample_rate, self.target_sr)
        
        # Normalize amplitude to [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Convert to float32
        audio_data = audio_data.astype(np.float32)
        
        # Apply light noise gate to reduce background noise
        audio_data = self._apply_noise_gate(audio_data)
        
        return audio_data
    
    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio_data: Input audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio data
        """
        try:
            import librosa
            resampled = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
            logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
            return resampled
        except ImportError:
            # Fallback: simple decimation/interpolation
            logger.warning("librosa not available, using simple resampling")
            ratio = target_sr / orig_sr
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            return resampled.astype(np.float32)
    
    def _apply_noise_gate(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Apply simple noise gate to reduce background noise
        
        Args:
            audio_data: Input audio data
            threshold: Amplitude threshold for noise gate
            
        Returns:
            Audio data with noise gate applied
        """
        # Simple noise gate: zero out samples below threshold
        mask = np.abs(audio_data) > threshold
        audio_data = audio_data * mask
        
        # Smooth transitions to avoid clicks
        kernel_size = int(0.001 * self.target_sr)  # 1ms smoothing
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            mask_smooth = np.convolve(mask.astype(float), kernel, mode='same')
            audio_data = audio_data * mask_smooth
        
        return audio_data
    
    def save_normalized_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save normalized audio to file
        
        Args:
            audio_data: Normalized audio data
            output_path: Path to save the audio file
        """
        self.file_utils.save_audio(audio_data, output_path, self.target_sr)
        logger.debug(f"Saved normalized audio to: {output_path}")
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get information about audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            audio_data, sample_rate = self.load_and_normalize(audio_path)
            
            duration = len(audio_data) / self.target_sr
            
            return {
                'file_path': audio_path,
                'duration_seconds': duration,
                'sample_rate': self.target_sr,
                'channels': 1,
                'format': 'float32',
                'samples': len(audio_data),
                'rms_level': float(np.sqrt(np.mean(audio_data ** 2))),
                'peak_level': float(np.max(np.abs(audio_data)))
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}
    
    def create_audio_stream(self, audio_data: np.ndarray, chunk_size: int = 16000):
        """
        Create streaming chunks of audio data
        
        Args:
            audio_data: Input audio data
            chunk_size: Size of each chunk in samples
            
        Yields:
            Audio chunks as numpy arrays
        """
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            yield chunk
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate if audio file can be processed
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            audio_path = Path(audio_path)
            
            # Check if file exists
            if not audio_path.exists():
                logger.error(f"File does not exist: {audio_path}")
                return False
            
            # Check file size (minimum 1KB)
            if audio_path.stat().st_size < 1024:
                logger.error(f"File too small: {audio_path}")
                return False
            
            # Check file extension
            ext = audio_path.suffix.lower()
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
            if ext not in supported_formats:
                logger.error(f"Unsupported format: {ext}")
                return False
            
            # Try to load a small portion
            if ext in ['.wav', '.flac']:
                try:
                    info = sf.info(str(audio_path))
                    if info.duration < 1.0:  # Minimum 1 second
                        logger.error(f"Audio too short: {info.duration}s")
                        return False
                except:
                    return False
            else:
                try:
                    audio_segment = AudioSegment.from_file(str(audio_path))
                    if len(audio_segment) < 1000:  # Minimum 1 second
                        logger.error(f"Audio too short: {len(audio_segment)}ms")
                        return False
                except:
                    return False
            
            logger.info(f"Audio file validation passed: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            return False


def process_audio_file(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convenience function to process an audio file
    
    Args:
        audio_path: Path to input audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (normalized_audio, sample_rate)
    """
    ingestion = AudioIngestion(target_sr)
    return ingestion.load_and_normalize(audio_path)
