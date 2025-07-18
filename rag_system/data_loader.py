"""
Data Loader for Audio RAG System

Handles loading and parsing of audio analysis data from the output/sessions/ directory.
Provides unified interface for accessing transcriptions, emotions, speakers, and semantic data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
from collections import Counter
import glob

class DataLoader:
    """
    Loads and parses audio analysis data from session directories.
    
    Supports multiple data types:
    - Transcription data (speech-to-text results)
    - Emotion detection (emotional analysis per segment) 
    - Speaker diarization (who spoke when)
    - Semantic segmentation (topics and themes)
    - Summarization (session summaries and key points)
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.sessions_dir = self.output_dir / "sessions"
        
        # Available data types and their expected file names
        self.data_types = {
            'transcription': 'transcription.json',
            'diarization': 'diarization.json',
            'emotion_detection': 'emotion_detection.json',
            'semantic_segmentation': 'semantic_segmentation.json',
            'summarization': 'summarization.json',
            'segment_enrichment': 'segment_enrichment.json',
            'final_report': 'final_report.json',
            'complete_results': 'complete_results.json'
        }
        
        # Cache for loaded data
        self._data_cache = {}
        self._cache_enabled = True
        
        # Statistics
        self.load_stats = {
            'sessions_loaded': 0,
            'files_loaded': 0,
            'errors': 0,
            'cache_hits': 0
        }
    
    def discover_sessions(self) -> List[str]:
        """
        Discover all available sessions in the output directory.
        
        Returns:
            List of session IDs found
        """
        if not self.sessions_dir.exists():
            return []
        
        sessions = []
        for session_path in self.sessions_dir.iterdir():
            if session_path.is_dir():
                sessions.append(session_path.name)
        
        return sorted(sessions)
    
    def get_session_files(self, session_id: str) -> Dict[str, bool]:
        """
        Check which data files are available for a session.
        
        Args:
            session_id: ID of the session to check
            
        Returns:
            Dictionary mapping data types to availability
        """
        session_path = self.sessions_dir / session_id
        availability = {}
        
        for data_type, filename in self.data_types.items():
            file_path = session_path / filename
            availability[data_type] = file_path.exists() and file_path.stat().st_size > 0
        
        return availability
    
    def load_session_data(self, 
                         session_id: str, 
                         data_types: Optional[List[str]] = None,
                         use_cache: bool = True) -> Dict[str, Any]:
        """
        Load all available data for a session.
        
        Args:
            session_id: ID of the session to load
            data_types: Specific data types to load (all if None)
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing loaded data by type
        """
        cache_key = f"{session_id}_{','.join(sorted(data_types)) if data_types else 'all'}"
        
        # Check cache first
        if use_cache and self._cache_enabled and cache_key in self._data_cache:
            self.load_stats['cache_hits'] += 1
            return self._data_cache[cache_key]
        
        session_path = self.sessions_dir / session_id
        if not session_path.exists():
            raise ValueError(f"Session directory not found: {session_id}")
        
        session_data = {}
        types_to_load = data_types or list(self.data_types.keys())
        
        for data_type in types_to_load:
            if data_type not in self.data_types:
                print(f"⚠️  Unknown data type: {data_type}")
                continue
            
            try:
                data = self._load_data_file(session_path, data_type)
                if data is not None:
                    session_data[data_type] = data
                    self.load_stats['files_loaded'] += 1
                    
            except Exception as e:
                print(f"❌ Error loading {data_type} for {session_id}: {e}")
                self.load_stats['errors'] += 1
        
        # Cache the result
        if self._cache_enabled:
            self._data_cache[cache_key] = session_data
        
        self.load_stats['sessions_loaded'] += 1
        return session_data
    
    def _load_data_file(self, session_path: Path, data_type: str) -> Optional[Dict]:
        """
        Load a specific data file type.
        
        Args:
            session_path: Path to the session directory
            data_type: Type of data to load
            
        Returns:
            Loaded data or None if file doesn't exist
        """
        filename = self.data_types[data_type]
        file_path = session_path / filename
        
        if not file_path.exists() or file_path.stat().st_size == 0:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate and normalize data structure
            return self._normalize_data(data, data_type)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading {filename}: {e}")
    
    def _normalize_data(self, data: Dict, data_type: str) -> Dict:
        """
        Normalize data structure to ensure consistency.
        
        Args:
            data: Raw data from JSON file
            data_type: Type of data being normalized
            
        Returns:
            Normalized data structure
        """
        if data_type == 'transcription':
            return self._normalize_transcription(data)
        elif data_type == 'diarization':
            return self._normalize_diarization(data)
        elif data_type == 'emotion_detection':
            return self._normalize_emotion_detection(data)
        elif data_type == 'semantic_segmentation':
            return self._normalize_semantic_segmentation(data)
        elif data_type == 'summarization':
            return self._normalize_summarization(data)
        else:
            # Return as-is for other types
            return data
    
    def _normalize_transcription(self, data: Dict) -> Dict:
        """Normalize transcription data structure."""
        normalized = {
            'segments': [],
            'metadata': data.get('metadata', {}),
            'total_duration': data.get('total_duration', 0),
            'language': data.get('language', 'unknown')
        }
        
        segments = data.get('segments', data.get('transcription', []))
        if isinstance(segments, list):
            for segment in segments:
                if isinstance(segment, dict):
                    normalized_segment = {
                        'start': float(segment.get('start', 0)),
                        'end': float(segment.get('end', 0)),
                        'text': str(segment.get('text', '')).strip(),
                        'speaker': segment.get('speaker', 'unknown'),
                        'confidence': float(segment.get('confidence', 0.0))
                    }
                    if normalized_segment['text']:
                        normalized['segments'].append(normalized_segment)
        
        return normalized
    
    def _normalize_diarization(self, data: Dict) -> Dict:
        """Normalize speaker diarization data."""
        normalized = {
            'speakers': {},
            'metadata': data.get('metadata', {}),
            'total_speakers': data.get('total_speakers', 0)
        }
        
        speakers_data = data.get('speakers', data.get('diarization', {}))
        if isinstance(speakers_data, dict):
            for speaker_id, speaker_info in speakers_data.items():
                normalized_speaker = {
                    'speaker_id': speaker_id,
                    'total_speaking_time': speaker_info.get('total_speaking_time', 0),
                    'segments': []
                }
                
                segments = speaker_info.get('segments', [])
                if isinstance(segments, list):
                    for segment in segments:
                        if isinstance(segment, dict):
                            normalized_segment = {
                                'start': float(segment.get('start', 0)),
                                'end': float(segment.get('end', 0)),
                                'duration': float(segment.get('duration', 0)),
                                'confidence': float(segment.get('confidence', 0.0)),
                                'text': segment.get('text', '')
                            }
                            normalized_speaker['segments'].append(normalized_segment)
                
                normalized['speakers'][speaker_id] = normalized_speaker
        
        return normalized
    
    def _normalize_emotion_detection(self, data: Dict) -> Dict:
        """Normalize emotion detection data."""
        normalized = {
            'emotions': [],
            'metadata': data.get('metadata', {}),
            'emotion_summary': data.get('emotion_summary', {})
        }
        
        emotions = data.get('emotions', data.get('emotion_detection', []))
        if isinstance(emotions, list):
            for emotion in emotions:
                if isinstance(emotion, dict):
                    normalized_emotion = {
                        'timestamp': float(emotion.get('timestamp', 0)),
                        'emotion': emotion.get('emotion', 'neutral'),
                        'confidence': float(emotion.get('confidence', 0.0)),
                        'intensity': float(emotion.get('intensity', 0.0)),
                        'text': emotion.get('text', ''),
                        'speaker': emotion.get('speaker', 'unknown')
                    }
                    normalized['emotions'].append(normalized_emotion)
        
        return normalized
    
    def _normalize_semantic_segmentation(self, data: Dict) -> Dict:
        """Normalize semantic segmentation data."""
        normalized = {
            'topics': [],
            'metadata': data.get('metadata', {}),
            'topic_summary': data.get('topic_summary', {})
        }
        
        topics = data.get('topics', data.get('semantic_segmentation', []))
        if isinstance(topics, list):
            for topic in topics:
                if isinstance(topic, dict):
                    normalized_topic = {
                        'start_time': float(topic.get('start_time', 0)),
                        'end_time': float(topic.get('end_time', 0)),
                        'topic': topic.get('topic', 'general'),
                        'confidence': float(topic.get('confidence', 0.0)),
                        'keywords': topic.get('keywords', []),
                        'content': topic.get('content', ''),
                        'importance': float(topic.get('importance', 0.0))
                    }
                    normalized['topics'].append(normalized_topic)
        
        return normalized
    
    def _normalize_summarization(self, data: Dict) -> Dict:
        """Normalize summarization data with rich emotional and contextual information."""
        normalized = {
            'blocks': [],
            'metadata': {},
            'global_insights': {},
            'emotional_landscape': {},
            'thematic_analysis': {}
        }
        
        # Handle list format (detailed blocks) vs dict format (simple summary)
        if isinstance(data, list):
            # Detailed block-by-block analysis
            all_themes = []
            all_sentiments = []
            all_emotions = {}
            total_segments = 0
            
            for block in data:
                if isinstance(block, dict):
                    block_data = {
                        'block_id': block.get('block_id', 0),
                        'start_time': block.get('start_time', 0),
                        'end_time': block.get('end_time', 0),
                        'duration': block.get('duration', 0),
                        'segment_count': block.get('segment_count', 0),
                        'text': block.get('text', ''),
                        'summary': block.get('summary', ''),
                        'key_points': block.get('key_points', []),
                        'insights': block.get('insights', {}),
                        'segments': []
                    }
                    
                    # Process segments with rich emotional data
                    segments = block.get('segments', [])
                    total_segments += len(segments)
                    
                    for segment in segments:
                        if isinstance(segment, dict):
                            # Extract emotional data
                            emotions = segment.get('emotions', {})
                            combined_emotion = emotions.get('combined_emotion', {})
                            
                            segment_data = {
                                'segment_id': segment.get('segment_id', 0),
                                'start_time': segment.get('start_time', 0),
                                'end_time': segment.get('end_time', 0),
                                'duration': segment.get('duration', 0),
                                'text': segment.get('text', ''),
                                'speaker': segment.get('speaker', 'unknown'),
                                'confidence': segment.get('confidence', 0.0),
                                'emotion': combined_emotion.get('emotion', 'neutral'),
                                'emotion_confidence': combined_emotion.get('confidence', 0.0),
                                'emotion_scores': combined_emotion.get('all_scores', {}),
                                'text_emotion': emotions.get('text_emotion', {}),
                                'audio_emotion': emotions.get('audio_emotion', {})
                            }
                            block_data['segments'].append(segment_data)
                            
                            # Aggregate emotions for analysis
                            emotion_name = combined_emotion.get('emotion', 'neutral')
                            if emotion_name in all_emotions:
                                all_emotions[emotion_name] += 1
                            else:
                                all_emotions[emotion_name] = 1
                    
                    # Extract insights
                    insights = block.get('insights', {})
                    if insights:
                        theme = insights.get('theme', '')
                        sentiment = insights.get('sentiment', '')
                        if theme:
                            all_themes.append(theme)
                        if sentiment:
                            all_sentiments.append(sentiment)
                    
                    normalized['blocks'].append(block_data)
            
            # Compile global insights
            normalized['global_insights'] = {
                'total_blocks': len(normalized['blocks']),
                'total_segments': total_segments,
                'dominant_themes': list(set(all_themes))[:10],
                'sentiment_distribution': dict(Counter(all_sentiments)),
                'overall_sentiment': Counter(all_sentiments).most_common(1)[0][0] if all_sentiments else 'neutral'
            }
            
            normalized['emotional_landscape'] = {
                'emotion_distribution': all_emotions,
                'dominant_emotion': max(all_emotions.items(), key=lambda x: x[1])[0] if all_emotions else 'neutral',
                'emotional_diversity': len(all_emotions),
                'total_emotion_points': sum(all_emotions.values())
            }
            
        else:
            # Simple dictionary format
            normalized.update({
                'summary': data.get('summary', ''),
                'key_points': data.get('key_points', []),
                'themes': data.get('themes', []),
                'keywords': data.get('keywords', []),
                'metadata': data.get('metadata', {}),
                'sentiment': data.get('sentiment', {}),
                'action_items': data.get('action_items', [])
            })
        
        return normalized
    
    def load_multiple_sessions(self, 
                              session_ids: List[str], 
                              data_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Load data for multiple sessions efficiently.
        
        Args:
            session_ids: List of session IDs to load
            data_types: Specific data types to load
            
        Returns:
            Dictionary mapping session IDs to their data
        """
        all_session_data = {}
        
        for session_id in session_ids:
            try:
                session_data = self.load_session_data(session_id, data_types)
                if session_data:
                    all_session_data[session_id] = session_data
                    
            except Exception as e:
                print(f"❌ Error loading session {session_id}: {e}")
        
        return all_session_data
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a quick summary of a session's available data.
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Summary dictionary
        """
        availability = self.get_session_files(session_id)
        session_path = self.sessions_dir / session_id
        
        summary = {
            'session_id': session_id,
            'available_files': [dtype for dtype, available in availability.items() if available],
            'file_count': sum(availability.values()),
            'session_exists': session_path.exists()
        }
        
        # Try to get basic metadata from final_report if available
        if availability.get('final_report'):
            try:
                report_data = self._load_data_file(session_path, 'final_report')
                if report_data:
                    summary.update({
                        'duration': report_data.get('total_duration_seconds', 0),
                        'audio_file': report_data.get('audio_file', 'unknown'),
                        'processing_time': report_data.get('processing_time', 0)
                    })
            except Exception:
                pass
        
        return summary
    
    def search_sessions_by_content(self, 
                                  search_term: str, 
                                  session_ids: Optional[List[str]] = None,
                                  data_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for sessions containing specific content.
        
        Args:
            search_term: Term to search for
            session_ids: Specific sessions to search (all if None)
            data_types: Data types to search in
            
        Returns:
            List of matches with context
        """
        sessions_to_search = session_ids or self.discover_sessions()
        types_to_search = data_types or ['transcription', 'summarization']
        matches = []
        
        search_term_lower = search_term.lower()
        
        for session_id in sessions_to_search:
            try:
                session_data = self.load_session_data(session_id, types_to_search)
                
                # Search in transcription
                if 'transcription' in session_data:
                    for segment in session_data['transcription'].get('segments', []):
                        text = segment.get('text', '').lower()
                        if search_term_lower in text:
                            matches.append({
                                'session_id': session_id,
                                'data_type': 'transcription',
                                'content': segment.get('text', ''),
                                'timestamp': segment.get('start', 0),
                                'speaker': segment.get('speaker', 'unknown'),
                                'match_type': 'transcription_segment'
                            })
                
                # Search in summarization
                if 'summarization' in session_data:
                    summary_data = session_data['summarization']
                    
                    # Check summary text
                    summary_text = summary_data.get('summary', '').lower()
                    if search_term_lower in summary_text:
                        matches.append({
                            'session_id': session_id,
                            'data_type': 'summarization',
                            'content': summary_data.get('summary', ''),
                            'match_type': 'summary'
                        })
                    
                    # Check key points
                    for point in summary_data.get('key_points', []):
                        if search_term_lower in str(point).lower():
                            matches.append({
                                'session_id': session_id,
                                'data_type': 'summarization',
                                'content': str(point),
                                'match_type': 'key_point'
                            })
                            
            except Exception as e:
                print(f"❌ Error searching session {session_id}: {e}")
        
        return matches
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get statistics about data loading operations."""
        return self.load_stats.copy()
    
    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        print("✅ Data cache cleared")
    
    def enable_cache(self):
        """Enable data caching."""
        self._cache_enabled = True
    
    def disable_cache(self):
        """Disable data caching."""
        self._cache_enabled = False
        self.clear_cache()

# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    print("🔍 Discovering sessions...")
    sessions = loader.discover_sessions()
    print(f"Found {len(sessions)} sessions: {sessions}")
    
    if sessions:
        # Test loading first session
        session_id = sessions[0]
        print(f"\n📊 Testing with session: {session_id}")
        
        # Check available files
        availability = loader.get_session_files(session_id)
        print(f"Available files: {[k for k, v in availability.items() if v]}")
        
        # Load session data
        try:
            data = loader.load_session_data(session_id)
            print(f"Loaded data types: {list(data.keys())}")
            
            # Show sample from each data type
            for data_type, content in data.items():
                if isinstance(content, dict):
                    if 'segments' in content:
                        print(f"  {data_type}: {len(content['segments'])} segments")
                    elif 'topics' in content:
                        print(f"  {data_type}: {len(content['topics'])} topics")
                    elif 'emotions' in content:
                        print(f"  {data_type}: {len(content['emotions'])} emotions")
                    else:
                        print(f"  {data_type}: {type(content)} data")
                        
        except Exception as e:
            print(f"❌ Error testing session loading: {e}")
    
    # Show statistics
    stats = loader.get_load_statistics()
    print(f"\n📈 Load Statistics: {stats}")
