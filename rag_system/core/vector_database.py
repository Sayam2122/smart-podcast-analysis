"""
Smart Vector Database for Audio RAG System

Multi-file vector database with hierarchical indexing, metadata filtering,
and comprehensive search capabilities for podcast analysis data.
"""

import chromadb
from chromadb.config import Settings
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer

from ..utils.data_loader import DataLoader
from ..utils.logger import PerformanceLogger

logger = logging.getLogger(__name__)


class SmartVectorDB:
    """
    Comprehensive vector database for multi-modal podcast analysis.
    
    Features:
    - Hierarchical indexing (Episode → Blocks → Segments)
    - Multi-layer embedding strategy
    - Metadata filtering and search
    - Cross-episode analysis capabilities
    """
    
    def __init__(self, 
                 db_path: str = "chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 sessions_path: str = "output/sessions"):
        """
        Initialize the Smart Vector Database.
        
        Args:
            db_path: Path to ChromaDB storage
            embedding_model: SentenceTransformer model name
            sessions_path: Path to podcast sessions data
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize data loader
        self.data_loader = DataLoader(sessions_path)
        
        # Initialize performance logger
        self.perf_logger = PerformanceLogger(logger)
        
        # Collection names for different content levels
        self.collections = {
            'episodes': None,      # Episode-level summaries
            'blocks': None,        # Block-level content
            'segments': None,      # Segment-level content
            'speakers': None,      # Speaker-specific content
            'emotions': None       # Emotion-tagged content
        }
        
        self._initialize_collections()
        
        # Session tracking
        self.loaded_sessions = set()
        self.session_metadata = {}
    
    def _initialize_collections(self):
        """Initialize all ChromaDB collections with proper metadata."""
        
        # Episode-level collection
        self.collections['episodes'] = self.client.get_or_create_collection(
            name="episodes",
            metadata={
                "description": "Episode-level summaries and metadata",
                "content_type": "episode_summary"
            }
        )
        
        # Block-level collection
        self.collections['blocks'] = self.client.get_or_create_collection(
            name="blocks",
            metadata={
                "description": "Content blocks with mid-level granularity",
                "content_type": "content_block"
            }
        )
        
        # Segment-level collection
        self.collections['segments'] = self.client.get_or_create_collection(
            name="segments",
            metadata={
                "description": "Individual segments with precise content",
                "content_type": "segment"
            }
        )
        
        # Speaker-specific collection
        self.collections['speakers'] = self.client.get_or_create_collection(
            name="speakers",
            metadata={
                "description": "Speaker-specific content and statistics",
                "content_type": "speaker_content"
            }
        )
        
        # Emotion-tagged collection
        self.collections['emotions'] = self.client.get_or_create_collection(
            name="emotions",
            metadata={
                "description": "Emotion-tagged content for affective search",
                "content_type": "emotion_content"
            }
        )
        
        logger.info("Initialized all vector database collections")
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a complete podcast session into the vector database.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        if session_id in self.loaded_sessions:
            logger.info(f"Session {session_id} already loaded")
            return True
        
        try:
            self.perf_logger.start_timer(f"load_session_{session_id}")
            
            # Load session data
            session_data = self.data_loader.load_session_data(session_id)
            
            # Store session metadata
            self.session_metadata[session_id] = session_data.get('metadata', {})
            
            # Load episode-level content
            self._load_episode_content(session_id, session_data)
            
            # Load block-level content
            self._load_block_content(session_id, session_data)
            
            # Load segment-level content
            self._load_segment_content(session_id, session_data)
            
            # Load speaker-specific content
            self._load_speaker_content(session_id, session_data)
            
            # Load emotion-tagged content
            self._load_emotion_content(session_id, session_data)
            
            self.loaded_sessions.add(session_id)
            
            duration = self.perf_logger.end_timer(f"load_session_{session_id}")
            logger.info(f"Successfully loaded session {session_id} in {duration:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def _load_episode_content(self, session_id: str, session_data: Dict):
        """Load episode-level summaries and metadata."""
        final_report = session_data.get('final_report', {})
        summaries = final_report.get('summaries', {})
        
        # Compile episode-level content for embedding
        episode_content = []
        
        # Add all summary content
        for key in ['overall_summary', 'global_summary']:
            if summaries.get(key):
                episode_content.append(summaries[key])
        
        # Add highlights and insights
        episode_content.extend(summaries.get('key_highlights', []))
        episode_content.extend(summaries.get('key_insights', []))
        episode_content.extend(summaries.get('main_topics', []))
        
        if episode_content:
            content_text = " ".join(episode_content)
            
            # Generate embedding
            embedding = self.embedding_model.encode(content_text).tolist()
            
            # Prepare metadata
            metadata = {
                'session_id': session_id,
                'content_type': 'episode',
                'created_at': datetime.now().isoformat(),
                'duration': final_report.get('audio_properties', {}).get('duration', 0),
                'total_speakers': final_report.get('content_analysis', {}).get('total_speakers', 0),
                'total_segments': final_report.get('content_analysis', {}).get('total_segments', 0),
                'quality_score': final_report.get('content_analysis', {}).get('quality_score', 0),
                'main_topics': json.dumps(summaries.get('main_topics', [])),
                'processing_date': final_report.get('session_metadata', {}).get('processing_date', '')
            }
            
            # Add to collection
            self.collections['episodes'].add(
                embeddings=[embedding],
                documents=[content_text],
                metadatas=[metadata],
                ids=[f"episode_{session_id}"]
            )
            
            self.perf_logger.increment_counter("episodes_loaded")
    
    def _load_block_content(self, session_id: str, session_data: Dict):
        """Load content blocks with mid-level granularity."""
        summarization = session_data.get('summarization', {})
        blocks = summarization.get('content_blocks', [])
        
        for block in blocks:
            block_id = block.get('block_id')
            if not block_id:
                continue
            
            # Compile block content
            block_content = []
            
            if block.get('summary'):
                block_content.append(block['summary'])
            
            block_content.extend(block.get('key_points', []))
            
            # Add segment summaries
            for segment in block.get('segments', []):
                if segment.get('summary'):
                    block_content.append(segment['summary'])
            
            if block_content:
                content_text = " ".join(block_content)
                
                # Generate embedding
                embedding = self.embedding_model.encode(content_text).tolist()
                
                # Prepare metadata
                metadata = {
                    'session_id': session_id,
                    'block_id': block_id,
                    'content_type': 'block',
                    'start_time': block.get('start_time', 0),
                    'end_time': block.get('end_time', 0),
                    'duration': block.get('duration', 0),
                    'themes': json.dumps(block.get('themes', [])),
                    'speaker_distribution': json.dumps(block.get('speaker_distribution', {})),
                    'confidence_score': block.get('confidence_score', 0),
                    'importance_score': block.get('importance_score', 0),
                    'segment_count': len(block.get('segments', []))
                }
                
                # Add emotion summary if available
                emotion_summary = block.get('emotion_summary', {})
                if emotion_summary:
                    metadata['dominant_emotion'] = emotion_summary.get('dominant_emotion', '')
                    metadata['emotion_intensity'] = emotion_summary.get('intensity', 0)
                
                # Add to collection
                self.collections['blocks'].add(
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[metadata],
                    ids=[f"block_{session_id}_{block_id}"]
                )
                
                self.perf_logger.increment_counter("blocks_loaded")
    
    def _load_segment_content(self, session_id: str, session_data: Dict):
        """Load individual segments with precise content."""
        all_segments = session_data.get('all_segments', [])
        
        for segment in all_segments:
            segment_id = segment.get('segment_id')
            if not segment_id:
                continue
            
            # Compile segment content
            segment_content = []
            
            if segment.get('text'):
                segment_content.append(segment['text'])
            
            if segment.get('summary'):
                segment_content.append(segment['summary'])
            
            segment_content.extend(segment.get('key_points', []))
            segment_content.extend(segment.get('keywords', []))
            
            if segment_content:
                content_text = " ".join(segment_content)
                
                # Generate embedding
                embedding = self.embedding_model.encode(content_text).tolist()
                
                # Prepare metadata
                metadata = {
                    'session_id': session_id,
                    'segment_id': segment_id,
                    'block_id': segment.get('block_id', ''),
                    'content_type': 'segment',
                    'start_time': segment.get('start_time', 0),
                    'end_time': segment.get('end_time', 0),
                    'duration': segment.get('duration', 0),
                    'speaker': segment.get('speaker', ''),
                    'speaker_confidence': segment.get('speaker_confidence', 0),
                    'confidence': segment.get('confidence', 0),
                    'importance': segment.get('importance', ''),
                    'sentiment': segment.get('sentiment', ''),
                    'themes': json.dumps(segment.get('themes', [])),
                    'topics': json.dumps(segment.get('topics', [])),
                    'keywords': json.dumps(segment.get('keywords', []))
                }
                
                # Add emotion data
                combined_emotion = segment.get('combined_emotion', {})
                if combined_emotion:
                    metadata['emotion'] = combined_emotion.get('emotion', '')
                    metadata['emotion_confidence'] = combined_emotion.get('confidence', 0)
                    metadata['emotion_intensity'] = combined_emotion.get('intensity', '')
                
                text_emotion = segment.get('text_emotion', {})
                if text_emotion:
                    metadata['text_emotion'] = text_emotion.get('emotion', '')
                    metadata['text_emotion_confidence'] = text_emotion.get('confidence', 0)
                
                audio_emotion = segment.get('audio_emotion', {})
                if audio_emotion:
                    metadata['audio_emotion'] = audio_emotion.get('emotion', '')
                    metadata['audio_emotion_confidence'] = audio_emotion.get('confidence', 0)
                
                # Add to collection
                self.collections['segments'].add(
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[metadata],
                    ids=[f"segment_{session_id}_{segment_id}"]
                )
                
                self.perf_logger.increment_counter("segments_loaded")
    
    def _load_speaker_content(self, session_id: str, session_data: Dict):
        """Load speaker-specific content aggregations."""
        speaker_map = session_data.get('speaker_map', {})
        final_report = session_data.get('final_report', {})
        speaker_stats = final_report.get('speaker_statistics', {})
        
        for speaker_id, segments in speaker_map.items():
            # Compile speaker-specific content
            speaker_content = []
            
            # Add all speaker segments
            for segment in segments:
                if segment.get('text'):
                    speaker_content.append(segment['text'])
                if segment.get('summary'):
                    speaker_content.append(segment['summary'])
            
            # Add speaker statistics
            stats = speaker_stats.get(speaker_id, {})
            if stats.get('key_topics'):
                speaker_content.extend(stats['key_topics'])
            
            if speaker_content:
                content_text = " ".join(speaker_content)
                
                # Generate embedding
                embedding = self.embedding_model.encode(content_text).tolist()
                
                # Prepare metadata
                metadata = {
                    'session_id': session_id,
                    'speaker_id': speaker_id,
                    'content_type': 'speaker',
                    'total_duration': stats.get('total_duration', 0),
                    'segment_count': stats.get('segment_count', 0),
                    'average_confidence': stats.get('average_confidence', 0),
                    'speaking_ratio': stats.get('speaking_ratio', 0),
                    'dominant_emotions': json.dumps(stats.get('dominant_emotions', [])),
                    'key_topics': json.dumps(stats.get('key_topics', [])),
                    'sentiment_distribution': json.dumps(stats.get('sentiment_distribution', {}))
                }
                
                # Add to collection
                self.collections['speakers'].add(
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[metadata],
                    ids=[f"speaker_{session_id}_{speaker_id}"]
                )
                
                self.perf_logger.increment_counter("speakers_loaded")
    
    def _load_emotion_content(self, session_id: str, session_data: Dict):
        """Load emotion-tagged content for affective search."""
        emotion_map = session_data.get('emotion_map', {})
        
        for emotion, segments in emotion_map.items():
            # Compile emotion-specific content
            emotion_content = []
            
            for segment in segments:
                if segment.get('text'):
                    emotion_content.append(segment['text'])
                if segment.get('summary'):
                    emotion_content.append(segment['summary'])
            
            if emotion_content:
                content_text = " ".join(emotion_content)
                
                # Generate embedding
                embedding = self.embedding_model.encode(content_text).tolist()
                
                # Calculate emotion statistics
                total_confidence = sum(
                    s.get('combined_emotion', {}).get('confidence', 0) 
                    for s in segments
                )
                avg_confidence = total_confidence / len(segments) if segments else 0
                
                # Prepare metadata
                metadata = {
                    'session_id': session_id,
                    'emotion': emotion,
                    'content_type': 'emotion',
                    'segment_count': len(segments),
                    'average_confidence': avg_confidence,
                    'total_duration': sum(s.get('duration', 0) for s in segments),
                    'speakers': json.dumps(list(set(
                        s.get('speaker', '') for s in segments if s.get('speaker')
                    ))),
                    'time_distribution': json.dumps([
                        s.get('start_time', 0) for s in segments
                    ])
                }
                
                # Add to collection
                self.collections['emotions'].add(
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[metadata],
                    ids=[f"emotion_{session_id}_{emotion}"]
                )
                
                self.perf_logger.increment_counter("emotions_loaded")
    
    def load_all_sessions(self) -> Dict[str, bool]:
        """
        Load all available sessions into the vector database.
        
        Returns:
            Dictionary mapping session IDs to success status
        """
        sessions = self.data_loader.discover_sessions()
        results = {}
        
        logger.info(f"Loading {len(sessions)} sessions into vector database")
        
        for session_id in sessions:
            results[session_id] = self.load_session(session_id)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Successfully loaded {successful}/{len(sessions)} sessions")
        
        return results
    
    def search(self, 
               query: str,
               content_types: List[str] = None,
               session_ids: List[str] = None,
               filters: Dict[str, Any] = None,
               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform intelligent search across multiple content types and sessions.
        
        Args:
            query: Search query text
            content_types: Types of content to search ['episode', 'block', 'segment', 'speaker', 'emotion']
            session_ids: Specific sessions to search (None for all)
            filters: Additional metadata filters
            limit: Maximum number of results
            
        Returns:
            List of search results with metadata and scores
        """
        if content_types is None:
            content_types = ['segment', 'block', 'episode']
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        for content_type in content_types:
            if content_type not in self.collections:
                continue
            
            collection = self.collections[content_type]
            
            # Prepare where clause for filtering
            where_clause = {}
            
            if session_ids:
                where_clause['session_id'] = {'$in': session_ids}
            
            if filters:
                where_clause.update(self._build_where_clause(filters, content_type))
            
            try:
                # Perform search
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause if where_clause else None,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Process results
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'content_type': content_type,
                        'query': query
                    }
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Search error in {content_type}: {e}")
                continue
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        self.perf_logger.increment_counter("searches_performed")
        
        return all_results[:limit]
    
    def _build_where_clause(self, filters: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where_clause = {}
        
        # Common filters
        if 'speaker' in filters:
            if content_type in ['segment', 'speaker']:
                where_clause['speaker'] = filters['speaker']
        
        if 'emotion' in filters:
            if content_type in ['segment', 'emotion']:
                where_clause['emotion'] = filters['emotion']
        
        if 'min_confidence' in filters:
            if content_type == 'segment':
                where_clause['confidence'] = {'$gte': filters['min_confidence']}
        
        if 'time_range' in filters:
            start_time, end_time = filters['time_range']
            if content_type in ['segment', 'block']:
                where_clause['start_time'] = {'$gte': start_time, '$lte': end_time}
        
        if 'importance' in filters:
            if content_type in ['segment', 'block']:
                where_clause['importance'] = filters['importance']
        
        return where_clause
    
    def get_session_statistics(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a session or all sessions.
        
        Args:
            session_id: Specific session ID (None for all sessions)
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_sessions': len(self.loaded_sessions),
            'collections': {},
            'session_details': {}
        }
        
        # Collection statistics
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats['collections'][name] = count
            except:
                stats['collections'][name] = 0
        
        # Session-specific statistics
        if session_id and session_id in self.loaded_sessions:
            session_data = self.data_loader.load_session_data(session_id)
            stats['session_details'][session_id] = {
                'segments': len(session_data.get('all_segments', [])),
                'speakers': len(session_data.get('speaker_map', {})),
                'emotions': len(session_data.get('emotion_map', {})),
                'duration': session_data.get('final_report', {}).get('audio_properties', {}).get('duration', 0)
            }
        elif not session_id:
            # All sessions
            for sid in self.loaded_sessions:
                try:
                    session_data = self.data_loader.load_session_data(sid)
                    stats['session_details'][sid] = {
                        'segments': len(session_data.get('all_segments', [])),
                        'speakers': len(session_data.get('speaker_map', {})),
                        'emotions': len(session_data.get('emotion_map', {})),
                        'duration': session_data.get('final_report', {}).get('audio_properties', {}).get('duration', 0)
                    }
                except:
                    continue
        
        return stats
    
    def clear_session(self, session_id: str) -> bool:
        """
        Remove a session from the vector database.
        
        Args:
            session_id: Session to remove
            
        Returns:
            Success status
        """
        try:
            # Remove from all collections
            for collection in self.collections.values():
                # Get all documents for this session
                results = collection.get(where={'session_id': session_id})
                if results['ids']:
                    collection.delete(ids=results['ids'])
            
            # Remove from tracking
            self.loaded_sessions.discard(session_id)
            self.session_metadata.pop(session_id, None)
            
            logger.info(f"Cleared session {session_id} from vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all data from the vector database.
        
        Returns:
            Success status
        """
        try:
            # Delete all collections
            for name in list(self.collections.keys()):
                try:
                    self.client.delete_collection(name)
                except:
                    pass
            
            # Reinitialize collections
            self._initialize_collections()
            
            # Clear tracking
            self.loaded_sessions.clear()
            self.session_metadata.clear()
            
            logger.info("Cleared all data from vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear vector database: {e}")
            return False
