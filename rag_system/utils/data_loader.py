"""
Data Loader for Smart Audio RAG System

Handles loading and parsing of podcast analysis data from JSON files
including final_report.json and summarization.json with comprehensive
field extraction and validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Comprehensive data loader for podcast analysis files.
    Handles final_report.json and summarization.json with full field extraction.
    """
    
    def __init__(self, base_path: str = "output/sessions"):
        """
        Initialize the data loader.
        
        Args:
            base_path: Path to the sessions directory containing podcast data
        """
        self.base_path = Path(base_path)
        self.sessions_cache = {}
        self.metadata_cache = {}
        
    def discover_sessions(self) -> List[str]:
        """
        Discover all available podcast sessions in the base directory.
        
        Returns:
            List of session identifiers
        """
        sessions = []
        
        if not self.base_path.exists():
            logger.warning(f"Sessions directory not found: {self.base_path}")
            return sessions
        
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                # Check if required files exist
                final_report = session_dir / "final_report.json"
                summarization = session_dir / "summarization.json"
                
                if final_report.exists() or summarization.exists():
                    sessions.append(session_dir.name)
                    logger.debug(f"Discovered session: {session_dir.name}")
        
        logger.info(f"Discovered {len(sessions)} podcast sessions")
        return sorted(sessions)
    
    def load_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Load comprehensive data for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Complete session data with all fields extracted
        """
        if session_id in self.sessions_cache:
            return self.sessions_cache[session_id]
        
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session not found: {session_id}")
        
        session_data = {
            'session_id': session_id,
            'session_path': str(session_path),
            'final_report': {},
            'summarization': {},
            'metadata': {},
            'segments': [],
            'speakers': {},
            'emotions': {},
            'content_hierarchy': {}
        }
        
        # Load final report
        final_report_path = session_path / "final_report.json"
        if final_report_path.exists():
            session_data['final_report'] = self._load_final_report(final_report_path)
        
        # Load summarization data
        summarization_path = session_path / "summarization.json"
        if summarization_path.exists():
            session_data['summarization'] = self._load_summarization(summarization_path)
        
        # Extract processed data
        session_data.update(self._extract_processed_data(session_data))
        
        # Cache the result
        self.sessions_cache[session_id] = session_data
        return session_data
    
    def _load_final_report(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse final_report.json with complete field extraction.
        
        Args:
            file_path: Path to final_report.json
            
        Returns:
            Parsed final report data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract key fields for easy access
            processed_data = {
                'raw_data': data,
                'session_metadata': self._extract_session_metadata(data),
                'audio_properties': self._extract_audio_properties(data),
                'content_analysis': self._extract_content_analysis(data),
                'summaries': self._extract_summaries(data),
                'performance_metrics': self._extract_performance_metrics(data),
                'speaker_statistics': self._extract_speaker_statistics(data),
                'emotion_distribution': self._extract_emotion_distribution(data)
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading final report {file_path}: {e}")
            return {}
    
    def _load_summarization(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse summarization.json with hierarchical content extraction.
        
        Args:
            file_path: Path to summarization.json
            
        Returns:
            Parsed summarization data with enhanced structure
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract hierarchical content structure
            processed_data = {
                'raw_data': data,
                'content_blocks': self._extract_content_blocks(data),
                'segment_analysis': self._extract_segment_analysis(data),
                'speaker_segments': self._extract_speaker_segments(data),
                'emotion_timeline': self._extract_emotion_timeline(data),
                'temporal_structure': self._extract_temporal_structure(data),
                'quality_metrics': self._extract_quality_metrics(data)
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading summarization {file_path}: {e}")
            return {}
    
    def _extract_session_metadata(self, data: Dict) -> Dict[str, Any]:
        """Extract session-level metadata from final report."""
        return {
            'session_id': data.get('session_id'),
            'processing_date': data.get('processing_date'),
            'audio_file': data.get('audio_file'),
            'created_at': data.get('created_at'),
            'processing_version': data.get('processing_version')
        }
    
    def _extract_audio_properties(self, data: Dict) -> Dict[str, Any]:
        """Extract audio file properties."""
        audio_info = data.get('audio_properties', {})
        return {
            'duration': audio_info.get('duration'),
            'sample_rate': audio_info.get('sample_rate'),
            'file_size': audio_info.get('file_size'),
            'format': audio_info.get('format'),
            'channels': audio_info.get('channels')
        }
    
    def _extract_content_analysis(self, data: Dict) -> Dict[str, Any]:
        """Extract content analysis summary."""
        return {
            'total_segments': data.get('total_segments'),
            'total_speakers': data.get('total_speakers'),
            'segment_distribution': data.get('segment_distribution', {}),
            'content_themes': data.get('content_themes', []),
            'quality_score': data.get('quality_score')
        }
    
    def _extract_summaries(self, data: Dict) -> Dict[str, Any]:
        """Extract all summary content for embeddings."""
        return {
            'overall_summary': data.get('overall_summary'),
            'global_summary': data.get('global_summary'),
            'key_highlights': data.get('key_highlights', []),
            'global_highlights': data.get('global_highlights', []),
            'key_insights': data.get('key_insights', []),
            'main_topics': data.get('main_topics', []),
            'conclusions': data.get('conclusions', [])
        }
    
    def _extract_performance_metrics(self, data: Dict) -> Dict[str, Any]:
        """Extract processing performance data."""
        performance = data.get('processing_performance', {})
        return {
            'total_processing_time': performance.get('total_processing_time'),
            'transcription_time': performance.get('transcription_time'),
            'diarization_time': performance.get('diarization_time'),
            'emotion_detection_time': performance.get('emotion_detection_time'),
            'summarization_time': performance.get('summarization_time'),
            'quality_scores': performance.get('quality_scores', {})
        }
    
    def _extract_speaker_statistics(self, data: Dict) -> Dict[str, Any]:
        """Extract comprehensive speaker statistics."""
        speakers = data.get('speaker_statistics', {})
        processed_speakers = {}
        
        for speaker_id, stats in speakers.items():
            processed_speakers[speaker_id] = {
                'total_duration': stats.get('total_duration'),
                'segment_count': stats.get('segment_count'),
                'average_confidence': stats.get('average_confidence'),
                'dominant_emotions': stats.get('dominant_emotions', []),
                'speaking_ratio': stats.get('speaking_ratio'),
                'key_topics': stats.get('key_topics', []),
                'sentiment_distribution': stats.get('sentiment_distribution', {})
            }
        
        return processed_speakers
    
    def _extract_emotion_distribution(self, data: Dict) -> Dict[str, Any]:
        """Extract emotion distribution and patterns."""
        emotions = data.get('emotion_distribution', {})
        return {
            'overall_distribution': emotions.get('overall_distribution', {}),
            'text_emotions': emotions.get('text_emotions', {}),
            'audio_emotions': emotions.get('audio_emotions', {}),
            'emotion_timeline': emotions.get('emotion_timeline', []),
            'dominant_emotions': emotions.get('dominant_emotions', []),
            'emotion_intensity': emotions.get('emotion_intensity', {})
        }
    
    def _extract_content_blocks(self, data: Dict) -> List[Dict[str, Any]]:
        """Extract hierarchical content blocks from summarization."""
        blocks = []
        
        # Handle different possible structures
        if 'blocks' in data:
            for block in data['blocks']:
                blocks.append(self._process_content_block(block))
        elif 'content_blocks' in data:
            for block in data['content_blocks']:
                blocks.append(self._process_content_block(block))
        elif 'data' in data and isinstance(data['data'], list):
            for block in data['data']:
                blocks.append(self._process_content_block(block))
        
        return blocks
    
    def _process_content_block(self, block: Dict) -> Dict[str, Any]:
        """Process individual content block with all metadata."""
        return {
            'block_id': block.get('block_id'),
            'summary': block.get('summary'),
            'key_points': block.get('key_points', []),
            'themes': block.get('themes', []),
            'segments': self._extract_block_segments(block),
            'start_time': block.get('start_time'),
            'end_time': block.get('end_time'),
            'duration': block.get('duration'),
            'speaker_distribution': block.get('speaker_distribution', {}),
            'emotion_summary': block.get('emotion_summary', {}),
            'confidence_score': block.get('confidence_score'),
            'importance_score': block.get('importance_score')
        }
    
    def _extract_block_segments(self, block: Dict) -> List[Dict[str, Any]]:
        """Extract segments within a content block."""
        segments = []
        
        if 'segments' in block:
            for segment in block['segments']:
                segments.append(self._process_segment(segment))
        
        return segments
    
    def _process_segment(self, segment: Dict) -> Dict[str, Any]:
        """Process individual segment with comprehensive metadata."""
        return {
            'segment_id': segment.get('segment_id'),
            'text': segment.get('text'),
            'summary': segment.get('summary'),
            'key_points': segment.get('key_points', []),
            'start_time': segment.get('start_time'),
            'end_time': segment.get('end_time'),
            'duration': segment.get('duration'),
            'speaker': segment.get('speaker'),
            'speaker_confidence': segment.get('speaker_confidence'),
            'text_emotion': segment.get('text_emotion', {}),
            'audio_emotion': segment.get('audio_emotion', {}),
            'combined_emotion': segment.get('combined_emotion', {}),
            'topics': segment.get('topics', []),
            'themes': segment.get('themes', []),
            'confidence': segment.get('confidence'),
            'importance': segment.get('importance'),
            'sentiment': segment.get('sentiment'),
            'keywords': segment.get('keywords', [])
        }
    
    def _extract_segment_analysis(self, data: Dict) -> Dict[str, Any]:
        """Extract segment-level analysis patterns."""
        return {
            'total_segments': data.get('total_segments'),
            'average_segment_duration': data.get('average_segment_duration'),
            'segment_quality_distribution': data.get('segment_quality_distribution', {}),
            'emotion_consistency': data.get('emotion_consistency'),
            'speaker_transitions': data.get('speaker_transitions', [])
        }
    
    def _extract_speaker_segments(self, data: Dict) -> Dict[str, List]:
        """Extract speaker-specific segment collections."""
        speaker_segments = {}
        
        # Process all blocks and segments to group by speaker
        for block in self._extract_content_blocks(data):
            for segment in block['segments']:
                speaker = segment.get('speaker')
                if speaker:
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append(segment)
        
        return speaker_segments
    
    def _extract_emotion_timeline(self, data: Dict) -> List[Dict[str, Any]]:
        """Extract temporal emotion progression."""
        timeline = []
        
        for block in self._extract_content_blocks(data):
            for segment in block['segments']:
                if segment.get('combined_emotion'):
                    timeline.append({
                        'timestamp': segment.get('start_time'),
                        'duration': segment.get('duration'),
                        'speaker': segment.get('speaker'),
                        'emotion': segment['combined_emotion'].get('emotion'),
                        'confidence': segment['combined_emotion'].get('confidence'),
                        'intensity': segment['combined_emotion'].get('intensity'),
                        'text_emotion': segment.get('text_emotion', {}),
                        'audio_emotion': segment.get('audio_emotion', {})
                    })
        
        return sorted(timeline, key=lambda x: x.get('timestamp', 0))
    
    def _extract_temporal_structure(self, data: Dict) -> Dict[str, Any]:
        """Extract temporal organization patterns."""
        blocks = self._extract_content_blocks(data)
        
        return {
            'total_duration': max((b.get('end_time', 0) for b in blocks), default=0),
            'block_count': len(blocks),
            'average_block_duration': sum(b.get('duration', 0) for b in blocks) / len(blocks) if blocks else 0,
            'time_distribution': self._calculate_time_distribution(blocks),
            'content_density': self._calculate_content_density(blocks)
        }
    
    def _calculate_time_distribution(self, blocks: List[Dict]) -> Dict[str, float]:
        """Calculate how content is distributed over time."""
        if not blocks:
            return {}
        
        total_duration = max((b.get('end_time', 0) for b in blocks), default=0)
        if total_duration == 0:
            return {}
        
        # Divide into quarters and calculate content distribution
        quarter_duration = total_duration / 4
        quarters = [0, 0, 0, 0]
        
        for block in blocks:
            start_time = block.get('start_time', 0)
            quarter_index = min(int(start_time // quarter_duration), 3)
            quarters[quarter_index] += block.get('duration', 0)
        
        return {
            'q1_ratio': quarters[0] / total_duration,
            'q2_ratio': quarters[1] / total_duration,
            'q3_ratio': quarters[2] / total_duration,
            'q4_ratio': quarters[3] / total_duration
        }
    
    def _calculate_content_density(self, blocks: List[Dict]) -> Dict[str, float]:
        """Calculate content density metrics."""
        if not blocks:
            return {}
        
        total_segments = sum(len(b.get('segments', [])) for b in blocks)
        total_duration = sum(b.get('duration', 0) for b in blocks)
        
        return {
            'segments_per_minute': total_segments / (total_duration / 60) if total_duration > 0 else 0,
            'average_segment_length': total_duration / total_segments if total_segments > 0 else 0,
            'content_coverage': total_duration  # Could be enhanced with silence detection
        }
    
    def _extract_quality_metrics(self, data: Dict) -> Dict[str, Any]:
        """Extract quality and confidence metrics."""
        return {
            'overall_confidence': data.get('overall_confidence'),
            'transcription_quality': data.get('transcription_quality'),
            'diarization_quality': data.get('diarization_quality'),
            'emotion_detection_quality': data.get('emotion_detection_quality'),
            'summary_compression_ratio': data.get('summary_compression_ratio'),
            'data_completeness': data.get('data_completeness', {})
        }
    
    def _extract_processed_data(self, session_data: Dict) -> Dict[str, Any]:
        """Extract and organize processed data for easy access."""
        processed = {
            'all_segments': [],
            'speaker_map': {},
            'emotion_map': {},
            'temporal_index': {},
            'content_index': {},
            'searchable_content': []
        }
        
        # Collect all segments with enhanced metadata
        if 'summarization' in session_data and 'content_blocks' in session_data['summarization']:
            for block in session_data['summarization']['content_blocks']:
                for segment in block.get('segments', []):
                    enhanced_segment = {
                        **segment,
                        'block_id': block.get('block_id'),
                        'block_summary': block.get('summary'),
                        'block_themes': block.get('themes', []),
                        'session_id': session_data['session_id']
                    }
                    processed['all_segments'].append(enhanced_segment)
        
        # Build indexes for efficient querying
        for segment in processed['all_segments']:
            # Speaker index
            speaker = segment.get('speaker')
            if speaker:
                if speaker not in processed['speaker_map']:
                    processed['speaker_map'][speaker] = []
                processed['speaker_map'][speaker].append(segment)
            
            # Emotion index
            emotion = segment.get('combined_emotion', {}).get('emotion')
            if emotion:
                if emotion not in processed['emotion_map']:
                    processed['emotion_map'][emotion] = []
                processed['emotion_map'][emotion].append(segment)
            
            # Temporal index (by minute)
            start_time = segment.get('start_time', 0)
            minute_key = int(start_time // 60)
            if minute_key not in processed['temporal_index']:
                processed['temporal_index'][minute_key] = []
            processed['temporal_index'][minute_key].append(segment)
            
            # Searchable content compilation
            searchable_text = " ".join(filter(None, [
                segment.get('text', ''),
                segment.get('summary', ''),
                " ".join(segment.get('key_points', [])),
                " ".join(segment.get('themes', [])),
                " ".join(segment.get('keywords', []))
            ]))
            
            if searchable_text.strip():
                processed['searchable_content'].append({
                    'content': searchable_text,
                    'segment_id': segment.get('segment_id'),
                    'metadata': segment
                })
        
        return processed
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a session for display.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary with key metrics
        """
        session_data = self.load_session_data(session_id)
        
        summary = {
            'session_id': session_id,
            'metadata': session_data.get('metadata', {}),
            'statistics': {
                'total_segments': len(session_data.get('all_segments', [])),
                'total_speakers': len(session_data.get('speaker_map', {})),
                'total_duration': 0,
                'emotion_distribution': {},
                'content_themes': []
            }
        }
        
        # Calculate statistics
        segments = session_data.get('all_segments', [])
        if segments:
            summary['statistics']['total_duration'] = max(
                s.get('end_time', 0) for s in segments
            )
            
            # Emotion distribution
            emotions = {}
            for segment in segments:
                emotion = segment.get('combined_emotion', {}).get('emotion')
                if emotion:
                    emotions[emotion] = emotions.get(emotion, 0) + 1
            summary['statistics']['emotion_distribution'] = emotions
            
            # Content themes
            themes = set()
            for segment in segments:
                themes.update(segment.get('themes', []))
            summary['statistics']['content_themes'] = list(themes)
        
        return summary
    
    def get_all_sessions_summary(self) -> List[Dict[str, Any]]:
        """
        Get summaries for all discovered sessions.
        
        Returns:
            List of session summaries
        """
        sessions = self.discover_sessions()
        summaries = []
        
        for session_id in sessions:
            try:
                summary = self.get_session_summary(session_id)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                continue
        
        return summaries
