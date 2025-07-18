"""
RAG Engine - Core functionality for Audio RAG System

Provides ChromaDB-powered vector search, embeddings, and LLM integration
for querying audio analysis data with intelligent context retrieval.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import traceback

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    print("⚠️  ChromaDB not installed. Install with: pip install chromadb")
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  SentenceTransformers not installed. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("⚠️  Requests not installed. Install with: pip install requests")
    REQUESTS_AVAILABLE = False

from data_loader import DataLoader
from context_router import ContextRouter

class RAGEngine:
    """
    Core RAG engine for audio analysis data.
    
    Provides vector-based search, semantic similarity, and LLM integration
    for intelligent querying of transcriptions, emotions, and speaker data.
    """
    
    def __init__(self, output_dir: str = "output", model_name: str = "all-MiniLM-L6-v2"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Initialize components
        self.data_loader = DataLoader(output_dir)
        self.context_router = ContextRouter()
        
        # ChromaDB setup
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        # Session tracking
        self.loaded_sessions = set()
        self.session_metadata = {}
        
        # Configuration
        self.ollama_base_url = "http://localhost:11434"
        self.max_context_length = 8000
        self.chunk_size = 500
        
        # Initialize if dependencies available
        if CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_components()
        else:
            print("⚠️  Some dependencies missing. Limited functionality available.")
    
    def _initialize_components(self):
        """Initialize ChromaDB and embedding model."""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.output_dir / "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="audio_rag",
                metadata={"description": "Audio analysis RAG collection"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            
            print(f"✅ RAG Engine initialized with {self.model_name}")
            
        except Exception as e:
            print(f"❌ Error initializing RAG engine: {e}")
            self.chroma_client = None
            self.collection = None
            self.embedding_model = None
    
    def load_sessions(self, session_ids: List[str], force_reload: bool = False):
        """
        Load sessions into the vector database.
        
        Args:
            session_ids: List of session IDs to load
            force_reload: Whether to reload already loaded sessions
        """
        if not self.collection or not self.embedding_model:
            print("❌ RAG engine not properly initialized")
            return
        
        sessions_to_load = []
        for session_id in session_ids:
            if session_id not in self.loaded_sessions or force_reload:
                sessions_to_load.append(session_id)
        
        if not sessions_to_load:
            print("✅ All requested sessions already loaded")
            return
        
        print(f"🔄 Loading {len(sessions_to_load)} sessions into vector database...")
        
        for session_id in sessions_to_load:
            try:
                self._load_session_data(session_id)
                self.loaded_sessions.add(session_id)
                print(f"  ✅ Loaded: {session_id}")
                
            except Exception as e:
                print(f"  ❌ Failed to load {session_id}: {e}")
        
        print(f"🎯 Total loaded sessions: {len(self.loaded_sessions)}")
    
    def _load_session_data(self, session_id: str):
        """Load data from a single session into ChromaDB."""
        # Load all available data for this session
        session_data = self.data_loader.load_session_data(session_id)
        
        if not session_data:
            raise ValueError(f"No data found for session {session_id}")
        
        documents = []
        metadatas = []
        ids = []
        
        # Process transcription data
        if 'transcription' in session_data:
            transcription = session_data['transcription']
            segments = transcription.get('segments', [])
            
            for i, segment in enumerate(segments):
                text = segment.get('text', '').strip()
                if not text:
                    continue
                
                doc_id = f"{session_id}_transcription_{i}"
                documents.append(text)
                
                metadata = {
                    'session_id': session_id,
                    'content_type': 'transcription',
                    'timestamp': segment.get('start', 0),
                    'duration': segment.get('end', 0) - segment.get('start', 0),
                    'speaker': segment.get('speaker', 'unknown'),
                    'confidence': segment.get('confidence', 0.0)
                }
                metadatas.append(metadata)
                ids.append(doc_id)
        
        # Process emotion data
        if 'emotion_detection' in session_data:
            emotions = session_data['emotion_detection']
            emotion_segments = emotions.get('emotions', [])
            
            for i, emotion_seg in enumerate(emotion_segments):
                text = emotion_seg.get('text', '').strip()
                if not text:
                    continue
                
                doc_id = f"{session_id}_emotion_{i}"
                documents.append(text)
                
                metadata = {
                    'session_id': session_id,
                    'content_type': 'emotion',
                    'timestamp': emotion_seg.get('timestamp', 0),
                    'emotion': emotion_seg.get('emotion', 'neutral'),
                    'confidence': emotion_seg.get('confidence', 0.0),
                    'intensity': emotion_seg.get('intensity', 0.0)
                }
                metadatas.append(metadata)
                ids.append(doc_id)
        
        # Process summarization data with rich emotional and contextual information
        if 'summarization' in session_data:
            summarization = session_data['summarization']
            
            # Handle both block format and simple format
            if 'blocks' in summarization:
                # Rich block-by-block data
                for block in summarization['blocks']:
                    # Add block-level summary
                    if block.get('summary'):
                        doc_id = f"{session_id}_summary_block_{block.get('block_id', 0)}"
                        documents.append(block['summary'])
                        
                        # Extract insights for enhanced metadata
                        insights = block.get('insights', {})
                        metadata = {
                            'session_id': session_id,
                            'content_type': 'summary_block',
                            'timestamp': block.get('start_time', 0),
                            'duration': block.get('duration', 0),
                            'block_id': block.get('block_id', 0),
                            'theme': insights.get('theme', 'general'),
                            'sentiment': insights.get('sentiment', 'neutral'),
                            'significance': insights.get('significance', ''),
                            'segment_count': block.get('segment_count', 0)
                        }
                        metadatas.append(metadata)
                        ids.append(doc_id)
                    
                    # Add key points
                    for i, key_point in enumerate(block.get('key_points', [])):
                        if key_point.strip():
                            doc_id = f"{session_id}_keypoint_block_{block.get('block_id', 0)}_{i}"
                            documents.append(key_point)
                            
                            metadata = {
                                'session_id': session_id,
                                'content_type': 'key_point',
                                'timestamp': block.get('start_time', 0),
                                'block_id': block.get('block_id', 0),
                                'theme': insights.get('theme', 'general'),
                                'sentiment': insights.get('sentiment', 'neutral'),
                                'importance': 'high'  # Key points are high importance
                            }
                            metadatas.append(metadata)
                            ids.append(doc_id)
                    
                    # Process segments with rich emotional data
                    for segment in block.get('segments', []):
                        text = segment.get('text', '').strip()
                        if text and len(text) > 10:  # Filter very short segments
                            doc_id = f"{session_id}_enriched_segment_{segment.get('segment_id', 0)}"
                            documents.append(text)
                            
                            metadata = {
                                'session_id': session_id,
                                'content_type': 'enriched_segment',
                                'timestamp': segment.get('start_time', 0),
                                'duration': segment.get('duration', 0),
                                'speaker': segment.get('speaker', 'unknown'),
                                'confidence': segment.get('confidence', 0.0),
                                'emotion': segment.get('emotion', 'neutral'),
                                'emotion_confidence': segment.get('emotion_confidence', 0.0),
                                'block_id': block.get('block_id', 0),
                                'theme': insights.get('theme', 'general'),
                                'sentiment': insights.get('sentiment', 'neutral')
                            }
                            
                            # Add detailed emotion scores if available
                            emotion_scores = segment.get('emotion_scores', {})
                            if emotion_scores:
                                # Find secondary emotions (confidence > 0.2)
                                secondary_emotions = [emotion for emotion, score in emotion_scores.items() 
                                                    if score > 0.2 and emotion != segment.get('emotion', 'neutral')]
                                metadata['secondary_emotions'] = ','.join(secondary_emotions[:3])
                            
                            metadatas.append(metadata)
                            ids.append(doc_id)
                
                # Add global insights if available
                global_insights = summarization.get('global_insights', {})
                if global_insights:
                    # Add dominant themes as searchable content
                    themes = global_insights.get('dominant_themes', [])
                    for i, theme in enumerate(themes[:5]):  # Top 5 themes
                        if theme.strip():
                            doc_id = f"{session_id}_global_theme_{i}"
                            documents.append(f"Main theme: {theme}")
                            
                            metadata = {
                                'session_id': session_id,
                                'content_type': 'global_theme',
                                'theme': theme,
                                'overall_sentiment': global_insights.get('overall_sentiment', 'neutral'),
                                'importance': 'high'
                            }
                            metadatas.append(metadata)
                            ids.append(doc_id)
            
            else:
                # Simple format - use existing logic
                summary_text = summarization.get('summary', '')
                if summary_text:
                    doc_id = f"{session_id}_summary"
                    documents.append(summary_text)
                    
                    metadata = {
                        'session_id': session_id,
                        'content_type': 'summary',
                        'importance': 'high'
                    }
                    metadatas.append(metadata)
                    ids.append(doc_id)
        
        # Process final report for context
        if 'final_report' in session_data:
            final_report = session_data['final_report']
            
            # Add key highlights
            for i, highlight in enumerate(final_report.get('key_highlights', [])):
                doc_id = f"{session_id}_highlight_{i}"
                documents.append(highlight)
                
                metadata = {
                    'session_id': session_id,
                    'content_type': 'highlight',
                    'importance': 'high',
                    'source': 'final_report'
                }
                metadatas.append(metadata)
                ids.append(doc_id)
            
            # Add overall summary
            overall_summary = final_report.get('overall_summary', '')
            if overall_summary:
                doc_id = f"{session_id}_overall_summary"
                documents.append(overall_summary)
                
                metadata = {
                    'session_id': session_id,
                    'content_type': 'overall_summary',
                    'duration': final_report.get('audio_info', {}).get('duration', 0),
                    'speakers_count': final_report.get('content_analysis', {}).get('speakers_detected', 0),
                    'importance': 'very_high'
                }
                metadatas.append(metadata)
                ids.append(doc_id)
        
        # Process diarization data
        if 'diarization' in session_data:
            diarization = session_data['diarization']
            speakers = diarization.get('speakers', {})
            
            for speaker_id, speaker_data in speakers.items():
                segments = speaker_data.get('segments', [])
                
                for i, segment in enumerate(segments):
                    text = segment.get('text', '').strip()
                    if not text:
                        continue
                    
                    doc_id = f"{session_id}_speaker_{speaker_id}_{i}"
                    documents.append(text)
                    
                    metadata = {
                        'session_id': session_id,
                        'content_type': 'speaker',
                        'timestamp': segment.get('start', 0),
                        'speaker': speaker_id,
                        'duration': segment.get('duration', 0),
                        'confidence': segment.get('confidence', 0.0)
                    }
                    metadatas.append(metadata)
                    ids.append(doc_id)
        
        # Add documents to ChromaDB in batches
        if documents:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            print(f"  📊 Added {len(documents)} documents from {session_id}")
        
        # Store session metadata
        self.session_metadata[session_id] = {
            'loaded_at': datetime.now().isoformat(),
            'document_count': len(documents),
            'data_types': list(session_data.keys())
        }
    
    def query(self, query_text: str, max_results: int = 10, min_similarity: float = 0.3) -> List[Dict]:
        """
        Perform semantic search across loaded sessions.
        
        Args:
            query_text: Natural language query
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching documents with metadata
        """
        if not self.collection or not self.embedding_model:
            print("❌ RAG engine not initialized")
            return []
        
        if not self.loaded_sessions:
            print("❌ No sessions loaded. Use load_sessions() first.")
            return []
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query_text],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            formatted_results = []
            for doc, meta, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score
                similarity = 1.0 - distance
                
                if similarity >= min_similarity:
                    result = {
                        'content': doc,
                        'similarity_score': similarity,
                        'session_id': meta.get('session_id'),
                        'content_type': meta.get('content_type'),
                        'timestamp': meta.get('timestamp'),
                        'speaker': meta.get('speaker'),
                        'emotion': meta.get('emotion'),
                        'topic': meta.get('topic'),
                        'confidence': meta.get('confidence', 0.0)
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def search_content(self, 
                      text_query: Optional[str] = None,
                      content_type: Optional[str] = None,
                      session_id: Optional[str] = None,
                      speaker: Optional[str] = None,
                      emotion: Optional[str] = None,
                      min_timestamp: Optional[float] = None,
                      max_timestamp: Optional[float] = None,
                      max_results: int = 20) -> List[Dict]:
        """
        Advanced search with multiple filters.
        
        Args:
            text_query: Text to search for
            content_type: Type of content (transcription, emotion, semantic, speaker)
            session_id: Specific session to search
            speaker: Specific speaker to filter by
            emotion: Specific emotion to filter by
            min_timestamp: Minimum timestamp
            max_timestamp: Maximum timestamp
            max_results: Maximum results to return
            
        Returns:
            List of matching documents
        """
        if not self.collection:
            print("❌ RAG engine not initialized")
            return []
        
        # Build where clause for filtering
        where_conditions = {}
        
        if content_type:
            where_conditions['content_type'] = content_type
        if session_id:
            where_conditions['session_id'] = session_id
        if speaker:
            where_conditions['speaker'] = speaker
        if emotion:
            where_conditions['emotion'] = emotion
        
        try:
            if text_query:
                # Semantic search with filters
                results = self.collection.query(
                    query_texts=[text_query],
                    n_results=max_results,
                    where=where_conditions if where_conditions else None,
                    include=['documents', 'metadatas', 'distances']
                )
                
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                formatted_results = []
                for doc, meta, distance in zip(documents, metadatas, distances):
                    # Apply timestamp filters
                    timestamp = meta.get('timestamp', 0)
                    if min_timestamp is not None and timestamp < min_timestamp:
                        continue
                    if max_timestamp is not None and timestamp > max_timestamp:
                        continue
                    
                    result = {
                        'content': doc,
                        'similarity_score': 1.0 - distance,
                        'session_id': meta.get('session_id'),
                        'content_type': meta.get('content_type'),
                        'timestamp': timestamp,
                        'speaker': meta.get('speaker'),
                        'emotion': meta.get('emotion'),
                        'topic': meta.get('topic'),
                        'confidence': meta.get('confidence', 0.0)
                    }
                    formatted_results.append(result)
                
                return formatted_results
            
            else:
                # Filter-only search
                results = self.collection.get(
                    where=where_conditions,
                    limit=max_results,
                    include=['documents', 'metadatas']
                )
                
                documents = results['documents']
                metadatas = results['metadatas']
                
                formatted_results = []
                for doc, meta in zip(documents, metadatas):
                    # Apply timestamp filters
                    timestamp = meta.get('timestamp', 0)
                    if min_timestamp is not None and timestamp < min_timestamp:
                        continue
                    if max_timestamp is not None and timestamp > max_timestamp:
                        continue
                    
                    result = {
                        'content': doc,
                        'similarity_score': 1.0,  # No semantic matching
                        'session_id': meta.get('session_id'),
                        'content_type': meta.get('content_type'),
                        'timestamp': timestamp,
                        'speaker': meta.get('speaker'),
                        'emotion': meta.get('emotion'),
                        'topic': meta.get('topic'),
                        'confidence': meta.get('confidence', 0.0)
                    }
                    formatted_results.append(result)
                
                return formatted_results
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def smart_query(self, query: str, max_results: int = 10) -> str:
        """
        LLM-enhanced query with context retrieval and intelligent response.
        
        Args:
            query: Natural language query
            max_results: Maximum context documents to retrieve
            
        Returns:
            LLM-generated response based on retrieved context
        """
        # First, get relevant context
        context_results = self.query(query, max_results=max_results)
        
        if not context_results:
            return "I couldn't find any relevant information in the loaded sessions for your query."
        
        # Prepare context for LLM
        context_text = self._prepare_context(context_results, query)
        
        # Generate LLM response
        return self._generate_llm_response(query, context_text)
    
    def _prepare_context(self, results: List[Dict], query: str) -> str:
        """Prepare context from search results for LLM."""
        context_parts = []
        
        # Group by session for better organization
        sessions = {}
        for result in results:
            session_id = result.get('session_id', 'unknown')
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(result)
        
        for session_id, session_results in sessions.items():
            context_parts.append(f"\n--- Session: {session_id} ---")
            
            for result in session_results:
                content_type = result.get('content_type', 'unknown')
                timestamp = result.get('timestamp', 0)
                speaker = result.get('speaker', '')
                emotion = result.get('emotion', '')
                
                context_line = f"[{content_type.upper()}]"
                if timestamp:
                    context_line += f" Time: {timestamp:.1f}s"
                if speaker:
                    context_line += f" Speaker: {speaker}"
                if emotion:
                    context_line += f" Emotion: {emotion}"
                
                context_line += f"\n{result['content']}\n"
                context_parts.append(context_line)
        
        return "\n".join(context_parts)
    
    def _generate_llm_response(self, query: str, context: str) -> str:
        """Generate response using Ollama LLM."""
        if not REQUESTS_AVAILABLE:
            return f"LLM functionality requires 'requests' library. Here's the retrieved context:\n\n{context}"
        
        # Prepare prompt
        prompt = f"""Based on the following audio analysis context, please answer the user's question.

Context from audio sessions:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. Focus on:
1. Direct answers to the question
2. Relevant details from the sessions
3. Speaker insights if applicable
4. Emotional context if relevant
5. Timeline information when useful

Answer:"""
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": "llama2",  # Default model, can be configured
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"LLM API error (status {response.status_code}). Context:\n\n{context}"
                
        except requests.exceptions.ConnectionError:
            return f"Cannot connect to Ollama (make sure it's running at {self.ollama_base_url}). Context:\n\n{context}"
        except Exception as e:
            return f"LLM error: {e}. Context:\n\n{context}"
    
    def get_loaded_sessions(self) -> List[str]:
        """Get list of currently loaded sessions."""
        return list(self.loaded_sessions)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded sessions."""
        if not self.collection:
            return {}
        
        try:
            # Get collection stats
            collection_count = self.collection.count()
            
            stats = {
                'loaded_sessions': len(self.loaded_sessions),
                'total_documents': collection_count,
                'sessions': list(self.loaded_sessions),
                'session_metadata': self.session_metadata
            }
            
            # Get content type distribution
            if collection_count > 0:
                # Sample to get content types
                sample_results = self.collection.get(limit=min(1000, collection_count))
                content_types = {}
                for meta in sample_results['metadatas']:
                    content_type = meta.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                
                stats['content_type_distribution'] = content_types
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the vector database."""
        if self.collection:
            try:
                # Delete the collection and recreate it
                self.chroma_client.delete_collection("audio_rag")
                self.collection = self.chroma_client.create_collection(
                    name="audio_rag",
                    metadata={"description": "Audio analysis RAG collection"}
                )
                self.loaded_sessions.clear()
                self.session_metadata.clear()
                print("✅ Collection cleared")
            except Exception as e:
                print(f"❌ Error clearing collection: {e}")

    def enhanced_smart_query(self, query: str, session_id: str = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Execute intelligent query with enhanced domain and emotion awareness.
        
        Args:
            query: Natural language question
            session_id: Session identifier (if None, searches all sessions)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with results, context, and metadata
        """
        try:
            # Get available data types
            available_types = self._get_available_data_types(session_id)
            
            # Route query with enhanced context awareness
            routing = self.context_router.route_query(query, available_types)
            
            # Extract routing information
            content_types = routing['content_types']
            filters = routing['filters']
            strategy = routing['strategy']
            domain = routing['domain']
            emotion_analysis = routing['emotion_analysis']
            
            # Search with strategy-based approach
            search_results = self._execute_search_strategy(
                query, session_id, content_types, filters, strategy, max_results
            )
            
            # Generate enhanced LLM prompt with domain and emotion context
            prompt = self._generate_enhanced_prompt(
                query, search_results, routing, domain, emotion_analysis
            )
            
            # Get LLM response
            llm_response = self._query_llm(prompt)
            
            # Return comprehensive result
            return {
                'query': query,
                'session_id': session_id,
                'domain': domain,
                'emotion_context': emotion_analysis,
                'answer': llm_response,
                'source_count': len(search_results),
                'sources': search_results,
                'routing_info': routing,
                'confidence': self._calculate_answer_confidence(search_results, llm_response, routing)
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced smart query failed: {e}")
            return {
                'query': query,
                'session_id': session_id,
                'domain': 'unknown',
                'emotion_context': {'primary_emotion': 'neutral', 'intensity': 'low'},
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'source_count': 0,
                'sources': [],
                'routing_info': {},
                'confidence': 0.0
            }
    
    def _execute_search_strategy(self, query: str, session_id: str, content_types: List[str], 
                               filters: Dict[str, Any], strategy: Dict[str, Any], max_results: int) -> List[Dict]:
        """Execute search based on strategy with enhanced filtering."""
        all_results = []
        
        # Adjust search parameters based on strategy
        search_type = strategy.get('search_type', 'semantic')
        emotion_weighting = strategy.get('emotion_weighting', 1.0)
        domain_boost = strategy.get('domain_boost', 1.0)
        
        for content_type in content_types:
            try:
                # Get type-specific results using existing search method
                if session_id:
                    # Search specific session
                    session_data = self.data_loader.load_session(session_id)
                    if content_type in session_data:
                        results = self._search_in_data(query, session_data[content_type], content_type)
                        # Apply filters
                        results = self._apply_filters(results, filters)
                        all_results.extend(results)
                else:
                    # Search all sessions
                    all_sessions = self.data_loader.list_sessions()
                    for sid in all_sessions:
                        session_data = self.data_loader.load_session(sid)
                        if content_type in session_data:
                            results = self._search_in_data(query, session_data[content_type], content_type)
                            results = self._apply_filters(results, filters)
                            all_results.extend(results)
                
            except Exception as e:
                self.logger.warning(f"Error searching {content_type}: {e}")
                continue
        
        # Apply strategy-specific scoring
        for result in all_results:
            result['strategy_score'] = self._calculate_strategy_score(
                result, strategy, emotion_weighting, domain_boost
            )
        
        # Sort by strategy score and return top results
        all_results.sort(key=lambda x: x.get('strategy_score', 0), reverse=True)
        return all_results[:max_results]
    
    def _apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply search filters to results."""
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Time filters
            if 'min_timestamp' in filters or 'max_timestamp' in filters:
                timestamp = metadata.get('start_time', metadata.get('timestamp', 0))
                if 'min_timestamp' in filters and timestamp < filters['min_timestamp']:
                    continue
                if 'max_timestamp' in filters and timestamp > filters['max_timestamp']:
                    continue
            
            # Speaker filter
            if 'speaker' in filters:
                if metadata.get('speaker', '').lower() != filters['speaker'].lower():
                    continue
            
            # Emotion filters
            if 'emotion' in filters:
                combined_emotion = metadata.get('combined_emotion', {})
                if combined_emotion.get('emotion', '').lower() != filters['emotion'].lower():
                    continue
            
            if 'emotion_confidence' in filters:
                combined_emotion = metadata.get('combined_emotion', {})
                if combined_emotion.get('confidence', 0) < filters['emotion_confidence']:
                    continue
            
            # Domain/theme filter
            if 'theme' in filters:
                themes = metadata.get('themes', [])
                if not any(filters['theme'].lower() in theme.lower() for theme in themes):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_strategy_score(self, result: Dict, strategy: Dict, emotion_weighting: float, domain_boost: float) -> float:
        """Calculate score based on search strategy and content analysis."""
        base_score = result.get('similarity_score', 0.5)
        
        # Apply emotion weighting if emotion data available
        if 'combined_emotion' in result.get('metadata', {}) and emotion_weighting > 1.0:
            emotion_confidence = result['metadata']['combined_emotion'].get('confidence', 0.5)
            base_score *= (1 + (emotion_weighting - 1) * emotion_confidence)
        
        # Apply domain boost if domain matches
        if domain_boost > 1.0 and 'themes' in result.get('metadata', {}):
            base_score *= domain_boost
        
        # Apply ranking factors
        for factor in strategy.get('ranking_factors', ['similarity']):
            if factor == 'importance' and 'significance' in result.get('metadata', {}):
                significance = result['metadata']['significance']
                if significance == 'high':
                    base_score *= 1.3
                elif significance == 'medium':
                    base_score *= 1.1
            elif factor == 'emotion_match' and 'combined_emotion' in result.get('metadata', {}):
                emotion_conf = result['metadata']['combined_emotion'].get('confidence', 0.5)
                base_score *= (1 + emotion_conf * 0.5)
        
        return base_score
    
    def _generate_enhanced_prompt(self, query: str, search_results: List[Dict], routing: Dict, 
                                domain: str, emotion_analysis: Dict) -> str:
        """Generate LLM prompt with domain and emotion context."""
        # Build context from search results
        context_pieces = []
        for result in search_results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Add emotional context if available
            emotion_info = ""
            if 'combined_emotion' in metadata:
                emotion = metadata['combined_emotion']
                emotion_info = f" [Emotion: {emotion.get('emotion', 'neutral')} (confidence: {emotion.get('confidence', 0.5):.2f})]"
            
            # Add speaker and timestamp info
            speaker_info = f" [Speaker: {metadata.get('speaker', 'Unknown')}]" if 'speaker' in metadata else ""
            time_info = f" [Time: {metadata.get('start_time', 0):.1f}s]" if 'start_time' in metadata else ""
            
            context_pieces.append(f"{content}{emotion_info}{speaker_info}{time_info}")
        
        context = "\n\n".join(context_pieces)
        
        # Build domain-aware prompt
        domain_instruction = ""
        if domain != 'general':
            domain_instruction = f"\n\nThis query is from the {domain} domain. Please provide insights relevant to this context."
        
        # Build emotion-aware prompt
        emotion_instruction = ""
        if emotion_analysis['primary_emotion'] != 'neutral':
            emotion_instruction = f"\n\nThe user seems to be seeking information with {emotion_analysis['primary_emotion']} emotional context (intensity: {emotion_analysis['intensity']}). Please be sensitive to this emotional aspect in your response."
        
        # Construct the full prompt
        prompt = f"""Based on the following context from a conversation session, please answer the user's question accurately and comprehensively.

Context:
{context}

User Question: {query}

{domain_instruction}{emotion_instruction}

Please provide a detailed, accurate answer based on the context provided. If the context doesn't fully answer the question, acknowledge what information is available and what might be missing.

Answer:"""

        return prompt
    
    def _calculate_answer_confidence(self, search_results: List[Dict], llm_response: str, routing: Dict) -> float:
        """Calculate confidence in the answer based on multiple factors."""
        confidence = 0.0
        
        # Base confidence from search results
        if search_results:
            avg_similarity = sum(r.get('similarity_score', 0) for r in search_results) / len(search_results)
            confidence += avg_similarity * 0.4
        
        # Confidence from routing quality
        routing_confidence = routing.get('routing_confidence', 0.5)
        confidence += routing_confidence * 0.2
        
        # Confidence from result count and diversity
        if len(search_results) >= 3:
            confidence += 0.2
        elif len(search_results) >= 1:
            confidence += 0.1
        
        # Confidence from emotion and domain analysis
        emotion_conf = routing.get('emotion_analysis', {}).get('confidence', 0.5)
        confidence += emotion_conf * 0.1
        
        # Confidence from answer length (longer usually means more comprehensive)
        if len(llm_response.split()) > 50:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_available_data_types(self, session_id: str = None) -> List[str]:
        """Get available data types for a session or all sessions."""
        if session_id:
            try:
                session_data = self.data_loader.load_session(session_id)
                return list(session_data.keys())
            except Exception:
                return []
        else:
            # Get all available types across all sessions
            available_types = set()
            try:
                all_sessions = self.data_loader.list_sessions()
                for sid in all_sessions:
                    session_data = self.data_loader.load_session(sid)
                    available_types.update(session_data.keys())
            except Exception:
                pass
            return list(available_types)
    
    def _search_in_data(self, query: str, data: List[Dict], content_type: str) -> List[Dict]:
        """Search within specific data type using semantic similarity."""
        if not data:
            return []
        
        results = []
        query_embedding = self.encoder.encode([query])
        
        for item in data:
            content = item.get('content', '')
            if not content:
                continue
            
            # Calculate similarity
            content_embedding = self.encoder.encode([content])
            similarity = self._calculate_similarity(query_embedding[0], content_embedding[0])
            
            result = {
                'content': content,
                'similarity_score': similarity,
                'content_type': content_type,
                'metadata': item
            }
            results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the RAG engine
    engine = RAGEngine()
    
    if engine.collection and engine.embedding_model:
        print("🎯 RAG Engine test mode")
        
        # Test session loading (if sessions exist)
        from session_manager import SessionManager
        manager = SessionManager()
        
        if manager.available_sessions:
            print(f"Testing with sessions: {list(manager.available_sessions)[:2]}")
            engine.load_sessions(list(manager.available_sessions)[:2])
            
            # Test basic query
            results = engine.query("What was discussed?", max_results=3)
            print(f"Query results: {len(results)}")
            
            # Test smart query (if Ollama available)
            # response = engine.smart_query("Summarize the main points")
            # print(f"Smart query response: {response[:100]}...")
    else:
        print("❌ RAG Engine test failed - dependencies missing")
