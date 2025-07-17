"""
Speaker diarization module for the podcast analysis pipeline.
Uses Pyannote-Audio for advanced speaker separation and clustering.
"""

import numpy as np
import torch
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class SpeakerDiarization:
    """
    Speaker diarization using Pyannote-Audio
    Performs speaker separation and clustering with confidence scores
    """
    
    def __init__(self, 
                 use_auth_token: Optional[str] = "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",
                 device: str = "auto",
                 num_speakers: Optional[int] = None,
                 min_speakers: int = 1,
                 max_speakers: int = 8):
        """
        Initialize speaker diarization
        
        Args:
            use_auth_token: Hugging Face auth token (optional)
            device: Device to use ("cpu", "cuda", "auto")
            num_speakers: Fixed number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        """
        self.use_auth_token = use_auth_token
        self.device = self._determine_device(device)
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None
        self.file_utils = get_file_utils()
        
        logger.info(f"Initializing speaker diarization | "
                   f"Device: {self.device} | "
                   f"Speakers: {num_speakers or f'{min_speakers}-{max_speakers}'}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            try:
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except:
                return "cpu"
        return device
    
    def _load_pipeline(self):
        """Load the Pyannote diarization pipeline"""
        if self.pipeline is not None:
            return
        
        start_time = time.time()
        
        try:
            from pyannote.audio import Pipeline
            
            # Load the pretrained pipeline
            if self.use_auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.use_auth_token
                )
            else:
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                except Exception:
                    # Fallback to older version or local model
                    logger.warning("Could not load latest model, trying fallback...")
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization"
                    )
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            
            load_time = time.time() - start_time
            logger.info(f"Loaded diarization pipeline in {load_time:.2f}s on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            logger.info("Falling back to simple speaker detection...")
            self.pipeline = "fallback"
    
    def diarize_audio(self, 
                     audio_data: np.ndarray, 
                     sample_rate: int = 16000) -> List[Dict]:
        """
        Perform speaker diarization on audio data
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            List of speaker segments with timestamps and labels
        """
        logger.info(f"Starting speaker diarization | "
                   f"Duration: {len(audio_data) / sample_rate:.2f}s")
        
        self._load_pipeline()
        
        start_time = time.time()
        
        try:
            if self.pipeline == "fallback":
                segments = self._fallback_diarization(audio_data, sample_rate)
            else:
                segments = self._pyannote_diarization(audio_data, sample_rate)
            
            # Post-process segments
            processed_segments = self._post_process_segments(segments)
            
            diarization_time = time.time() - start_time
            unique_speakers = len(set([s['speaker'] for s in processed_segments]))
            
            logger.info(f"Diarization completed | "
                       f"Segments: {len(processed_segments)} | "
                       f"Speakers: {unique_speakers} | "
                       f"Time: {diarization_time:.2f}s")
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # Return fallback result
            return self._fallback_diarization(audio_data, sample_rate)
    
    def _pyannote_diarization(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """Perform diarization using Pyannote"""
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save audio
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Set diarization parameters
            if self.num_speakers:
                diarization = self.pipeline(temp_path, num_speakers=self.num_speakers)
            else:
                diarization = self.pipeline(
                    temp_path,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers
                )
            
            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'confidence': 1.0  # Pyannote doesn't provide confidence scores directly
                })
            
            return segments
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
    
    def _fallback_diarization(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Fallback diarization using simple energy-based voice activity detection
        """
        logger.warning("Using fallback diarization method")
        
        # Simple VAD based on energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        frame_shift = int(0.010 * sample_rate)   # 10ms shift
        
        segments = []
        current_speaker = "Speaker 1"
        segment_start = 0.0
        
        for i in range(0, len(audio_data) - frame_length, frame_shift):
            frame = audio_data[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            
            # Simple speaker change detection based on energy patterns
            # This is very basic and not accurate, just for fallback
            time_pos = i / sample_rate
            
            if i == 0 or (i % (sample_rate * 10)) == 0:  # Change speaker every 10 seconds
                if segments:
                    segments[-1]['end'] = time_pos
                
                if time_pos > 0:
                    current_speaker = f"Speaker {(len(segments) % 2) + 1}"
                    segment_start = time_pos
        
        # Add segments
        duration = len(audio_data) / sample_rate
        if not segments:
            segments.append({
                'start': 0.0,
                'end': duration,
                'speaker': 'Speaker 1',
                'confidence': 0.5
            })
        else:
            segments[-1]['end'] = duration
        
        return segments
    
    def _post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """Post-process diarization segments"""
        if not segments:
            return []
        
        processed_segments = []
        
        for i, segment in enumerate(segments):
            # Skip very short segments (< 0.5 seconds)
            duration = segment['end'] - segment['start']
            if duration < 0.5:
                continue
            
            processed_segment = {
                'segment_id': i + 1,
                'start_time': round(segment['start'], 2),
                'end_time': round(segment['end'], 2),
                'duration': round(duration, 2),
                'speaker': segment['speaker'],
                'confidence': segment.get('confidence', 1.0)
            }
            
            processed_segments.append(processed_segment)
        
        # Merge consecutive segments from the same speaker
        processed_segments = self._merge_consecutive_speakers(processed_segments)
        
        # Re-number segments
        for i, segment in enumerate(processed_segments):
            segment['segment_id'] = i + 1
        
        return processed_segments
    
    def _merge_consecutive_speakers(self, segments: List[Dict], gap_threshold: float = 0.5) -> List[Dict]:
        """Merge consecutive segments from the same speaker"""
        if not segments:
            return segments
        
        merged_segments = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # Check if same speaker and small gap
            gap = next_segment['start_time'] - current_segment['end_time']
            
            if (current_segment['speaker'] == next_segment['speaker'] and 
                gap <= gap_threshold):
                
                # Merge segments
                current_segment['end_time'] = next_segment['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                current_segment['confidence'] = min(current_segment['confidence'], 
                                                   next_segment['confidence'])
            else:
                # Keep current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def align_with_transcript(self, 
                             diarization_segments: List[Dict], 
                             transcript_segments: List[Dict]) -> List[Dict]:
        """
        Align diarization with transcript segments
        
        Args:
            diarization_segments: Speaker diarization segments
            transcript_segments: Transcript segments
            
        Returns:
            Transcript segments with speaker labels
        """
        logger.info("Aligning diarization with transcript")
        
        aligned_segments = []
        
        for transcript_seg in transcript_segments:
            # Find the best matching speaker segment
            best_speaker = "Unknown"
            best_overlap = 0.0
            best_confidence = 0.0
            
            for diar_seg in diarization_segments:
                # Calculate overlap
                overlap_start = max(transcript_seg['start_time'], diar_seg['start_time'])
                overlap_end = min(transcript_seg['end_time'], diar_seg['end_time'])
                overlap = max(0, overlap_end - overlap_start)
                
                # Calculate overlap ratio
                transcript_duration = transcript_seg['end_time'] - transcript_seg['start_time']
                overlap_ratio = overlap / transcript_duration if transcript_duration > 0 else 0
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_speaker = diar_seg['speaker']
                    best_confidence = diar_seg['confidence']
            
            # Add speaker information to transcript segment
            aligned_segment = transcript_seg.copy()
            aligned_segment['speaker'] = best_speaker
            aligned_segment['speaker_confidence'] = best_confidence
            aligned_segment['overlap_ratio'] = best_overlap
            
            aligned_segments.append(aligned_segment)
        
        logger.info(f"Aligned {len(aligned_segments)} transcript segments with speakers")
        return aligned_segments
    
    def save_diarization(self, segments: List[Dict], output_path: str) -> None:
        """
        Save diarization results to JSON file
        
        Args:
            segments: Diarization segments
            output_path: Path to save the diarization results
        """
        diarization_data = {
            'metadata': {
                'num_speakers': len(set([s['speaker'] for s in segments])),
                'total_segments': len(segments),
                'total_duration': max([s['end_time'] for s in segments]) if segments else 0,
                'device': self.device
            },
            'segments': segments
        }
        
        self.file_utils.save_json(diarization_data, output_path)
        logger.info(f"Saved diarization to: {output_path}")
    
    def get_speaker_stats(self, segments: List[Dict]) -> Dict:
        """
        Get statistics about speakers
        
        Args:
            segments: Diarization segments
            
        Returns:
            Dictionary with speaker statistics
        """
        if not segments:
            return {}
        
        speaker_times = {}
        total_duration = 0
        
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']
            
            if speaker not in speaker_times:
                speaker_times[speaker] = 0
            speaker_times[speaker] += duration
            total_duration += duration
        
        # Calculate percentages
        speaker_percentages = {
            speaker: (time / total_duration) * 100 
            for speaker, time in speaker_times.items()
        }
        
        return {
            'num_speakers': len(speaker_times),
            'speaker_times': speaker_times,
            'speaker_percentages': speaker_percentages,
            'total_duration': total_duration,
            'average_segment_duration': np.mean([s['duration'] for s in segments]),
            'dominant_speaker': max(speaker_percentages, key=speaker_percentages.get)
        }
    
    def create_rttm_format(self, segments: List[Dict], audio_file: str = "audio") -> str:
        """
        Create RTTM format output for diarization
        
        Args:
            segments: Diarization segments
            audio_file: Audio file identifier
            
        Returns:
            RTTM format string
        """
        rttm_lines = []
        
        for segment in segments:
            # RTTM format: SPEAKER <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf>
            line = (f"SPEAKER {audio_file} 1 {segment['start_time']:.3f} "
                   f"{segment['duration']:.3f} <NA> <NA> {segment['speaker']} "
                   f"{segment['confidence']:.3f}")
            rttm_lines.append(line)
        
        return "\n".join(rttm_lines)


def diarize_audio_file(audio_path: str, 
                      num_speakers: Optional[int] = None,
                      use_auth_token: Optional[str] = None) -> List[Dict]:
    """
    Convenience function to diarize an audio file
    
    Args:
        audio_path: Path to audio file
        num_speakers: Number of speakers (if known)
        use_auth_token: Hugging Face auth token
        
    Returns:
        List of diarization segments
    """
    diarizer = SpeakerDiarization(
        num_speakers=num_speakers,
        use_auth_token=use_auth_token
    )
    
    from pipeline.audio_ingestion import AudioIngestion
    
    # Load and normalize audio
    ingestion = AudioIngestion()
    audio_data, sample_rate = ingestion.load_and_normalize(audio_path)
    
    # Diarize
    return diarizer.diarize_audio(audio_data, sample_rate)
