"""
Vector database manager for the podcast RAG system.
Handles embedding storage and retrieval using ChromaDB.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import uuid
from datetime import datetime

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class VectorDatabase:
    """
    Vector database manager using ChromaDB for podcast content storage and retrieval
    """
    
    def __init__(self,
                 db_path: str = "rag_system/vector_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "podcast_content"):
        """
        Initialize vector database
        
        Args:
            db_path: Path to ChromaDB storage
            embedding_model: SentenceTransformers model for embeddings
            collection_name: Name of the ChromaDB collection
        """
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        self.file_utils = get_file_utils()
        
        # Initialize database
        self._initialize_database()
        self._load_embedding_model()
        
        logger.info(f"Vector database initialized | Path: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Podcast content embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
        except ImportError:
            logger.error("ChromaDB not installed. Please install: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load sentence embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
        except ImportError:
            logger.error("SentenceTransformers not installed. Please install: pip install sentence-transformers")
            self.embedding_model = None
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def add_podcast_session(self, session_data: Dict) -> str:
        """
        Add a complete podcast session to the vector database
        
        Args:
            session_data: Complete session results from pipeline
            
        Returns:
            Session ID in the database
        """
        logger.info("Adding podcast session to vector database")
        
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return None
        
        session_id = session_data.get('session_info', {}).get('session_id', str(uuid.uuid4()))
        
        # Extract and index different content types
        added_count = 0
        
        # 1. Index semantic blocks with summaries
        if 'summaries' in session_data:
            added_count += self._index_semantic_blocks(session_data['summaries'], session_id)
        
        # 2. Index individual segments with detailed metadata
        if 'emotion_analysis' in session_data:
            added_count += self._index_segments(session_data['emotion_analysis'], session_id)
        
        # 3. Index overall summary and insights
        if 'final_report' in session_data:
            added_count += self._index_overall_summary(session_data['final_report'], session_id)
        
        logger.info(f"Added {added_count} embeddings for session {session_id}")
        return session_id
    
    def _index_semantic_blocks(self, blocks: List[Dict], session_id: str) -> int:
        """Index semantic blocks with summaries"""
        documents = []
        metadatas = []
        ids = []
        
        for block in blocks:
            if not block.get('summary'):
                continue
            
            # Create document from summary and key points
            doc_parts = [block['summary']]
            if block.get('key_points'):
                doc_parts.extend(block['key_points'])
            document = ' '.join(doc_parts)
            
            # Create metadata
            metadata = {
                'type': 'semantic_block',
                'session_id': session_id,
                'block_id': block.get('block_id', 0),
                'start_time': block.get('start_time', 0),
                'end_time': block.get('end_time', 0),
                'duration': block.get('duration', 0),
                'segment_count': block.get('segment_count', 0),
                'key_topics': ','.join(block.get('key_topics', [])),
                'insights_theme': block.get('insights', {}).get('theme', ''),
                'insights_sentiment': block.get('insights', {}).get('sentiment', 'neutral'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Create unique ID
            doc_id = f"{session_id}_block_{block.get('block_id', 0)}"
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        if documents:
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            logger.info(f"Indexed {len(documents)} semantic blocks")
        
        return len(documents)
    
    def _index_segments(self, segments: List[Dict], session_id: str) -> int:
        """Index individual segments with detailed metadata"""
        documents = []
        metadatas = []
        ids = []
        
        for i, segment in enumerate(segments):
            if not segment.get('text') or len(segment['text'].strip()) < 10:
                continue
            
            document = segment['text'].strip()
            
            # Create metadata
            metadata = {
                'type': 'segment',
                'session_id': session_id,
                'segment_index': i,
                'start_time': segment.get('start_time', 0),
                'end_time': segment.get('end_time', 0),
                'duration': segment.get('end_time', 0) - segment.get('start_time', 0),
                'speaker': segment.get('speaker', 'Unknown'),
                'speaker_confidence': segment.get('speaker_confidence', 0.0),
                'confidence': segment.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add emotion data if available
            if 'emotion' in segment:
                emotion_data = segment['emotion']
                metadata.update({
                    'emotion_label': emotion_data.get('label', 'neutral'),
                    'emotion_confidence': emotion_data.get('confidence', 0.0),
                    'emotion_source': emotion_data.get('source', 'unknown')
                })
            
            # Create unique ID
            doc_id = f"{session_id}_segment_{i}"
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        if documents:
            # Generate embeddings in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                embeddings = self.embedding_model.encode(batch_docs).tolist()
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings,
                    ids=batch_ids
                )
            
            logger.info(f"Indexed {len(documents)} segments")
        
        return len(documents)
    
    def _index_overall_summary(self, final_report: Dict, session_id: str) -> int:
        """Index overall summary and insights"""
        documents = []
        metadatas = []
        ids = []
        
        # Index overall summary
        if final_report.get('overall_summary'):
            document = final_report['overall_summary']
            
            metadata = {
                'type': 'overall_summary',
                'session_id': session_id,
                'audio_duration': final_report.get('audio_info', {}).get('duration', 0),
                'total_segments': final_report.get('content_analysis', {}).get('total_segments', 0),
                'semantic_blocks': final_report.get('content_analysis', {}).get('semantic_blocks', 0),
                'speakers_detected': final_report.get('content_analysis', {}).get('speakers_detected', 0),
                'processing_date': final_report.get('session_info', {}).get('processing_date', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(f"{session_id}_overall_summary")
        
        # Index key insights
        if final_report.get('key_insights'):
            insights_text = ' '.join(final_report['key_insights'])
            
            metadata = {
                'type': 'key_insights',
                'session_id': session_id,
                'insight_count': len(final_report['key_insights']),
                'timestamp': datetime.now().isoformat()
            }
            
            documents.append(insights_text)
            metadatas.append(metadata)
            ids.append(f"{session_id}_key_insights")
        
        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            logger.info(f"Indexed overall summary and insights")
        
        return len(documents)
    
    def search(self,
               query: str,
               n_results: int = 10,
               filters: Optional[Dict] = None,
               content_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for relevant content using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filters: Metadata filters
            content_types: Filter by content type (semantic_block, segment, overall_summary, key_insights)
            
        Returns:
            List of search results with documents and metadata
        """
        if not self.embedding_model:
            logger.error("Embedding model not available for search")
            return []
        
        logger.info(f"Searching for: '{query}' | Results: {n_results}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build filters
        where_filter = {}
        if filters:
            where_filter.update(filters)
        
        if content_types:
            where_filter["type"] = {"$in": content_types}
        
        try:
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:  # Check if results exist
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_speaker(self, speaker: str, n_results: int = 10) -> List[Dict]:
        """Search for content by specific speaker"""
        return self.search(
            query="",  # Empty query for metadata-only search
            n_results=n_results,
            filters={"speaker": speaker},
            content_types=["segment"]
        )
    
    def search_by_emotion(self, emotion: str, n_results: int = 10) -> List[Dict]:
        """Search for content by emotion"""
        return self.search(
            query="",
            n_results=n_results,
            filters={"emotion_label": emotion},
            content_types=["segment"]
        )
    
    def search_by_timeframe(self, start_time: float, end_time: float, n_results: int = 10) -> List[Dict]:
        """Search for content within a specific timeframe"""
        # Note: ChromaDB doesn't support range queries in metadata filters directly
        # This is a simplified implementation
        all_results = self.search(
            query="",
            n_results=1000,  # Get many results to filter
            content_types=["segment", "semantic_block"]
        )
        
        # Filter by timeframe
        filtered_results = []
        for result in all_results:
            result_start = result['metadata'].get('start_time', 0)
            result_end = result['metadata'].get('end_time', 0)
            
            # Check for overlap
            if (result_start <= end_time and result_end >= start_time):
                filtered_results.append(result)
            
            if len(filtered_results) >= n_results:
                break
        
        return filtered_results
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary information for a specific session"""
        try:
            # Get overall summary
            overall_results = self.collection.query(
                query_embeddings=[],
                n_results=1,
                where={"session_id": session_id, "type": "overall_summary"},
                include=["documents", "metadatas"]
            )
            
            # Get session statistics
            all_session_results = self.collection.query(
                query_embeddings=[],
                n_results=1000,
                where={"session_id": session_id},
                include=["metadatas"]
            )
            
            # Count content types
            content_counts = {}
            total_duration = 0
            speakers = set()
            
            if all_session_results['metadatas']:
                for metadata in all_session_results['metadatas'][0]:
                    content_type = metadata.get('type', 'unknown')
                    content_counts[content_type] = content_counts.get(content_type, 0) + 1
                    
                    if 'duration' in metadata:
                        total_duration += metadata['duration']
                    
                    if 'speaker' in metadata:
                        speakers.add(metadata['speaker'])
            
            summary = {
                'session_id': session_id,
                'overall_summary': overall_results['documents'][0][0] if overall_results['documents'] and overall_results['documents'][0] else None,
                'content_counts': content_counts,
                'total_duration': total_duration,
                'speakers': list(speakers),
                'total_embeddings': sum(content_counts.values())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {}
    
    def list_sessions(self) -> List[Dict]:
        """List all sessions in the database"""
        try:
            # Get all overall summaries (one per session)
            results = self.collection.query(
                query_embeddings=[],
                n_results=1000,
                where={"type": "overall_summary"},
                include=["metadatas"]
            )
            
            sessions = []
            if results['metadatas']:
                for metadata in results['metadatas'][0]:
                    session_info = {
                        'session_id': metadata.get('session_id'),
                        'processing_date': metadata.get('processing_date'),
                        'audio_duration': metadata.get('audio_duration', 0),
                        'total_segments': metadata.get('total_segments', 0),
                        'semantic_blocks': metadata.get('semantic_blocks', 0),
                        'speakers_detected': metadata.get('speakers_detected', 0)
                    }
                    sessions.append(session_info)
            
            return sorted(sessions, key=lambda x: x.get('processing_date', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all data for a specific session"""
        try:
            # Get all IDs for this session
            results = self.collection.query(
                query_embeddings=[],
                n_results=10000,
                where={"session_id": session_id},
                include=["metadatas"]
            )
            
            if results['ids'] and results['ids'][0]:
                ids_to_delete = results['ids'][0]
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} embeddings for session {session_id}")
                return True
            else:
                logger.warning(f"No data found for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            # Get collection info
            collection_info = self.collection.count()
            
            # Get content type distribution
            all_results = self.collection.query(
                query_embeddings=[],
                n_results=10000,
                include=["metadatas"]
            )
            
            content_types = {}
            sessions = set()
            
            if all_results['metadatas']:
                for metadata in all_results['metadatas'][0]:
                    content_type = metadata.get('type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    if 'session_id' in metadata:
                        sessions.add(metadata['session_id'])
            
            return {
                'total_embeddings': collection_info,
                'total_sessions': len(sessions),
                'content_type_distribution': content_types,
                'database_path': str(self.db_path),
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def reset_database(self) -> bool:
        """Reset the entire database (USE WITH CAUTION)"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Podcast content embeddings"}
            )
            logger.warning("Database reset completed")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False
