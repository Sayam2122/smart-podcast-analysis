"""
Transcription module for the podcast analysis pipeline.
Uses Whisper or Faster-Whisper for high-quality speech-to-text conversion.
"""

import numpy as np
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class Transcription:
    """
    Speech-to-text transcription using Whisper models
    Supports both openai-whisper and faster-whisper backends
    """
    
    def __init__(self, 
                 model_size: str = "base",
                 backend: str = "faster-whisper",
                 device: str = "auto",
                 compute_type: str = "float16"):
        """
        Initialize transcription module
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            backend: "whisper" or "faster-whisper"
            device: Device to use ("cpu", "cuda", "auto")
            compute_type: Compute type for faster-whisper
        """
        self.model_size = model_size
        self.backend = backend
        self.device = self._determine_device(device)
        self.compute_type = compute_type
        self.model = None
        self.file_utils = get_file_utils()
        
        logger.info(f"Initializing transcription with {backend} | "
                   f"Model: {model_size} | Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the Whisper model"""
        if self.model is not None:
            return
        
        start_time = time.time()
        
        try:
            if self.backend == "faster-whisper":
                from faster_whisper import WhisperModel
                
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                
            else:  # openai-whisper
                import whisper
                
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device
                )
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {self.backend} model '{self.model_size}' "
                       f"in {load_time:.2f}s on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_audio(self, 
                        audio_data: np.ndarray, 
                        sample_rate: int = 16000,
                        max_chunk_duration: float = 300.0) -> List[Dict]:
        """
        Transcribe audio data to text with timestamps
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            max_chunk_duration: Maximum duration per chunk in seconds (5 minutes default)
            
        Returns:
            List of transcript segments with start, end, and text
        """
        duration = len(audio_data) / sample_rate
        logger.info(f"Starting transcription | "
                   f"Duration: {duration:.2f}s | "
                   f"Model: {self.model_size}")
        
        self._load_model()
        
        start_time = time.time()
        
        try:
            # If audio is longer than max_chunk_duration, split into chunks
            if duration > max_chunk_duration:
                logger.info(f"Audio is {duration:.2f}s, splitting into chunks of {max_chunk_duration}s")
                segments = self._transcribe_chunked(audio_data, sample_rate, max_chunk_duration)
            else:
                if self.backend == "faster-whisper":
                    segments = self._transcribe_faster_whisper(audio_data, sample_rate)
                else:
                    segments = self._transcribe_openai_whisper(audio_data, sample_rate)
            
            # Post-process segments
            processed_segments = self._post_process_segments(segments)
            
            transcription_time = time.time() - start_time
            rtf = transcription_time / duration  # Real-time factor
            
            logger.info(f"Transcription completed | "
                       f"Segments: {len(processed_segments)} | "
                       f"Time: {transcription_time:.2f}s | "
                       f"RTF: {rtf:.2f}x")
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _transcribe_chunked(self, audio_data: np.ndarray, sample_rate: int, chunk_duration: float) -> List[Dict]:
        """Transcribe audio in chunks to manage memory usage"""
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(2.0 * sample_rate)  # 2 second overlap
        
        all_segments = []
        audio_offset = 0.0
        for i in range(0, len(audio_data), chunk_samples - overlap_samples):
            chunk_start = i
            chunk_end = min(i + chunk_samples, len(audio_data))
            chunk_audio = audio_data[chunk_start:chunk_end]
            if len(chunk_audio) < sample_rate:
                break
            logger.info(f"Processing chunk {i//chunk_samples + 1}: {chunk_start/sample_rate:.1f}s - {chunk_end/sample_rate:.1f}s")
            if self.backend == "faster-whisper":
                chunk_segments = self._transcribe_faster_whisper(chunk_audio, sample_rate)
            else:
                chunk_segments = self._transcribe_openai_whisper(chunk_audio, sample_rate)
            # Convert raw segments to post-processed format
            processed_chunk_segments = self._post_process_segments(chunk_segments)
            # Adjust timestamps to absolute time
            for segment in processed_chunk_segments:
                segment['start_time'] += audio_offset
                segment['end_time'] += audio_offset
                all_segments.append(segment)
            audio_offset += (chunk_samples - overlap_samples) / sample_rate
        return all_segments
    
    def _transcribe_faster_whisper(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """Transcribe using faster-whisper"""
        segments = []
        
        # faster-whisper expects audio as float32
        audio_float32 = audio_data.astype(np.float32)
        
        # Transcribe with word-level timestamps
        segments_generator, info = self.model.transcribe(
            audio_float32,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        logger.debug(f"Detected language: {info.language} "
                    f"(probability: {info.language_probability:.2f})")
        
        for segment in segments_generator:
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'no_speech_prob': getattr(segment, 'no_speech_prob', 0.0),
                'avg_logprob': getattr(segment, 'avg_logprob', 0.0),
                'compression_ratio': getattr(segment, 'compression_ratio', 0.0)
            })
        
        return segments
    
    def _transcribe_openai_whisper(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """Transcribe using openai-whisper"""
        # Save audio to temporary file for whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save audio
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Transcribe
            result = self.model.transcribe(
                temp_path,
                verbose=False,
                word_timestamps=False  # We don't need word-level for this pipeline
            )
            
            segments = []
            for segment in result['segments']:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'no_speech_prob': segment.get('no_speech_prob', 0.0),
                    'avg_logprob': segment.get('avg_logprob', 0.0),
                    'compression_ratio': segment.get('compression_ratio', 0.0)
                })
            
            return segments
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
    
    def _post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """Post-process transcript segments"""
        processed_segments = []
        
        for i, segment in enumerate(segments):
            # Clean up text
            text = segment['text'].strip()
            
            # Skip very short or empty segments
            if len(text) < 3:
                continue
            
            # Skip segments with very high no-speech probability
            if segment.get('no_speech_prob', 0.0) > 0.8:
                continue
            
            # Add segment ID
            processed_segment = {
                'segment_id': i + 1,
                'start_time': round(segment['start'], 2),
                'end_time': round(segment['end'], 2),
                'duration': round(segment['end'] - segment['start'], 2),
                'text': text,
                'confidence': 1.0 - segment.get('no_speech_prob', 0.0),
                'avg_logprob': segment.get('avg_logprob', 0.0),
                'compression_ratio': segment.get('compression_ratio', 0.0)
            }
            
            processed_segments.append(processed_segment)
        
        # Merge very short adjacent segments
        processed_segments = self._merge_short_segments(processed_segments)
        
        # Re-number segments after merging
        for i, segment in enumerate(processed_segments):
            segment['segment_id'] = i + 1
        
        return processed_segments
    
    def _merge_short_segments(self, segments: List[Dict], min_duration: float = 2.0) -> List[Dict]:
        """Merge segments that are too short"""
        if not segments:
            return segments
        
        merged_segments = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # If current segment is too short and gap is small, merge
            if (current_segment['duration'] < min_duration and 
                next_segment['start_time'] - current_segment['end_time'] < 1.0):
                
                # Merge segments
                current_segment['end_time'] = next_segment['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                current_segment['text'] += " " + next_segment['text']
                current_segment['confidence'] = min(current_segment['confidence'], 
                                                   next_segment['confidence'])
            else:
                # Keep current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def transcribe_file(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio file directly
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcript segments
        """
        from pipeline.audio_ingestion import AudioIngestion
        
        # Load and normalize audio
        ingestion = AudioIngestion()
        audio_data, sample_rate = ingestion.load_and_normalize(audio_path)
        
        # Transcribe
        return self.transcribe_audio(audio_data, sample_rate)
    
    def save_transcript(self, segments: List[Dict], output_path: str) -> None:
        """
        Save transcript to JSON file
        
        Args:
            segments: Transcript segments
            output_path: Path to save the transcript
        """
        transcript_data = {
            'metadata': {
                'model': self.model_size,
                'backend': self.backend,
                'device': self.device,
                'total_segments': len(segments),
                'total_duration': max([s['end_time'] for s in segments]) if segments else 0
            },
            'segments': segments
        }
        
        self.file_utils.save_json(transcript_data, output_path)
        logger.info(f"Saved transcript to: {output_path}")
    
    def get_full_text(self, segments: List[Dict]) -> str:
        """
        Get full transcript text
        
        Args:
            segments: Transcript segments
            
        Returns:
            Complete transcript as string
        """
        return " ".join([segment['text'] for segment in segments])
    
    def get_transcript_stats(self, segments: List[Dict]) -> Dict:
        """
        Get statistics about the transcript
        
        Args:
            segments: Transcript segments
            
        Returns:
            Dictionary with transcript statistics
        """
        if not segments:
            return {}
        
        total_duration = max([s['end_time'] for s in segments])
        total_words = sum([len(s['text'].split()) for s in segments])
        avg_confidence = np.mean([s['confidence'] for s in segments])
        
        return {
            'total_segments': len(segments),
            'total_duration': total_duration,
            'total_words': total_words,
            'words_per_minute': (total_words / total_duration) * 60 if total_duration > 0 else 0,
            'average_confidence': avg_confidence,
            'average_segment_duration': np.mean([s['duration'] for s in segments]),
            'shortest_segment': min([s['duration'] for s in segments]),
            'longest_segment': max([s['duration'] for s in segments])
        }
    
    def create_srt_subtitle(self, segments: List[Dict]) -> str:
        """
        Create SRT format subtitle string
        
        Args:
            segments: Transcript segments
            
        Returns:
            SRT format subtitle string
        """
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment['start_time'])
            end_time = self._seconds_to_srt_time(segment['end_time'])
            
            srt_lines.extend([
                str(i),
                f"{start_time} --> {end_time}",
                segment['text'],
                ""
            ])
        
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_audio_file(audio_path: str, 
                         model_size: str = "base",
                         backend: str = "faster-whisper") -> List[Dict]:
    """
    Convenience function to transcribe an audio file
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        backend: Transcription backend
        
    Returns:
        List of transcript segments
    """
    transcriber = Transcription(model_size=model_size, backend=backend)
    return transcriber.transcribe_file(audio_path)
