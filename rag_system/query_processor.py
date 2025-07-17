"""
Query processor for the podcast RAG system.
Handles natural language queries and retrieval using the vector database.
"""

import time
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from utils.logger import get_logger
from rag_system.vector_database import VectorDatabase

logger = get_logger(__name__)


class QueryProcessor:
    """
    Natural language query processor for podcast content retrieval
    Supports filtering, ranking, and contextual search
    """
    
    def __init__(self,
                 vector_db: Optional[VectorDatabase] = None,
                 max_results: int = 10,
                 min_similarity: float = 0.3):
        """
        Initialize query processor
        
        Args:
            vector_db: Vector database instance
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold
        """
        self.vector_db = vector_db or VectorDatabase()
        self.max_results = max_results
        self.min_similarity = min_similarity
        
        # Query patterns for intent detection
        self.query_patterns = self._initialize_query_patterns()
        
        logger.info("Query processor initialized")
    
    def _initialize_query_patterns(self) -> Dict:
        """Initialize patterns for query intent detection"""
        return {
            'speaker_queries': [
                r'what did (.+?) say',
                r'(.+?) said',
                r'who said (.+)',
                r'speaker (.+)',
                r'by (.+?) speaker'
            ],
            'emotion_queries': [
                r'(happy|sad|angry|excited|calm|neutral|surprised|fearful) moments',
                r'emotional (.+)',
                r'when (.+?) was (happy|sad|angry|excited|calm|neutral|surprised|fearful)',
                r'emotion (.+)'
            ],
            'time_queries': [
                r'at (\d+):(\d+)',
                r'between (\d+):(\d+) and (\d+):(\d+)',
                r'after (\d+) minutes',
                r'before (\d+) minutes',
                r'around (\d+):(\d+)'
            ],
            'topic_queries': [
                r'about (.+)',
                r'discussing (.+)',
                r'mentioned (.+)',
                r'topic (.+)',
                r'talk about (.+)'
            ],
            'summary_queries': [
                r'summarize',
                r'summary',
                r'main points',
                r'key takeaways',
                r'overview'
            ]
        }
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a natural language query and return relevant results
        
        Args:
            query: Natural language query
            session_id: Specific session to search (optional)
            
        Returns:
            Dictionary with query results and metadata
        """
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        # Parse query intent and extract parameters
        query_info = self._parse_query(query)
        
        # Build search parameters
        search_params = self._build_search_parameters(query_info, session_id)
        
        # Execute search
        results = self._execute_search(search_params)
        
        # Post-process and rank results
        processed_results = self._post_process_results(results, query_info)
        
        # Generate response
        response = self._generate_response(processed_results, query_info, query)
        
        processing_time = time.time() - start_time
        
        return {
            'query': query,
            'query_info': query_info,
            'results': processed_results,
            'response': response,
            'metadata': {
                'processing_time': processing_time,
                'total_results': len(processed_results),
                'session_filter': session_id,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _parse_query(self, query: str) -> Dict:
        """Parse query to extract intent and parameters"""
        query_lower = query.lower().strip()
        
        query_info = {
            'intent': 'general_search',
            'entities': {
                'speakers': [],
                'emotions': [],
                'topics': [],
                'time_references': []
            },
            'filters': {},
            'original_query': query,
            'cleaned_query': query_lower
        }
        
        # Check for speaker queries
        for pattern in self.query_patterns['speaker_queries']:
            match = re.search(pattern, query_lower)
            if match:
                query_info['intent'] = 'speaker_search'
                speaker = match.group(1).strip()
                query_info['entities']['speakers'].append(speaker)
                query_info['filters']['speaker'] = speaker
                break
        
        # Check for emotion queries
        for pattern in self.query_patterns['emotion_queries']:
            match = re.search(pattern, query_lower)
            if match:
                query_info['intent'] = 'emotion_search'
                # Extract emotion from the pattern
                emotions = ['happy', 'sad', 'angry', 'excited', 'calm', 'neutral', 'surprised', 'fearful']
                for emotion in emotions:
                    if emotion in match.group(0):
                        query_info['entities']['emotions'].append(emotion)
                        query_info['filters']['emotion_label'] = emotion
                        break
                break
        
        # Check for time queries
        for pattern in self.query_patterns['time_queries']:
            match = re.search(pattern, query_lower)
            if match:
                query_info['intent'] = 'time_search'
                time_ref = match.group(0)
                query_info['entities']['time_references'].append(time_ref)
                # Parse time (simplified - you could enhance this)
                if 'at' in time_ref:
                    time_parts = re.findall(r'(\d+):(\d+)', time_ref)
                    if time_parts:
                        minutes, seconds = time_parts[0]
                        target_time = int(minutes) * 60 + int(seconds)
                        query_info['filters']['target_time'] = target_time
                break
        
        # Check for topic queries
        for pattern in self.query_patterns['topic_queries']:
            match = re.search(pattern, query_lower)
            if match:
                if query_info['intent'] == 'general_search':  # Don't override more specific intents
                    query_info['intent'] = 'topic_search'
                topic = match.group(1).strip()
                query_info['entities']['topics'].append(topic)
                # Use the topic as the main search query
                query_info['cleaned_query'] = topic
                break
        
        # Check for summary queries
        for pattern in self.query_patterns['summary_queries']:
            if re.search(pattern, query_lower):
                query_info['intent'] = 'summary_request'
                break
        
        return query_info
    
    def _build_search_parameters(self, query_info: Dict, session_id: Optional[str]) -> Dict:
        """Build search parameters based on query analysis"""
        params = {
            'query': query_info['cleaned_query'],
            'n_results': self.max_results,
            'filters': {},
            'content_types': None
        }
        
        # Add session filter
        if session_id:
            params['filters']['session_id'] = session_id
        
        # Add filters based on intent
        if query_info['intent'] == 'speaker_search':
            params['filters'].update(query_info['filters'])
            params['content_types'] = ['segment']
        
        elif query_info['intent'] == 'emotion_search':
            params['filters'].update(query_info['filters'])
            params['content_types'] = ['segment']
        
        elif query_info['intent'] == 'summary_request':
            params['content_types'] = ['overall_summary', 'semantic_block', 'key_insights']
            params['n_results'] = 5  # Fewer results for summaries
        
        elif query_info['intent'] == 'topic_search':
            params['content_types'] = ['semantic_block', 'segment']
        
        # General search includes all content types
        
        return params
    
    def _execute_search(self, search_params: Dict) -> List[Dict]:
        """Execute search with the given parameters"""
        try:
            results = self.vector_db.search(**search_params)
            
            # Filter by minimum similarity
            filtered_results = [
                result for result in results 
                if result.get('similarity', 0) >= self.min_similarity
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []
    
    def _post_process_results(self, results: List[Dict], query_info: Dict) -> List[Dict]:
        """Post-process and rank results"""
        if not results:
            return results
        
        # Add query-specific scoring
        for result in results:
            result['score'] = self._calculate_relevance_score(result, query_info)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add result ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        # Format timestamps for display
        for result in results:
            metadata = result.get('metadata', {})
            if 'start_time' in metadata and 'end_time' in metadata:
                start_time = metadata['start_time']
                end_time = metadata['end_time']
                result['time_range'] = self._format_time_range(start_time, end_time)
        
        return results
    
    def _calculate_relevance_score(self, result: Dict, query_info: Dict) -> float:
        """Calculate relevance score for ranking"""
        # Base score from similarity
        base_score = result.get('similarity', 0)
        
        # Bonus for intent-specific matches
        metadata = result.get('metadata', {})
        bonus = 0
        
        # Speaker match bonus
        if query_info['intent'] == 'speaker_search':
            if any(speaker.lower() in metadata.get('speaker', '').lower() 
                   for speaker in query_info['entities']['speakers']):
                bonus += 0.2
        
        # Emotion match bonus
        if query_info['intent'] == 'emotion_search':
            if any(emotion.lower() == metadata.get('emotion_label', '').lower()
                   for emotion in query_info['entities']['emotions']):
                bonus += 0.3
        
        # Content type preference
        content_type = metadata.get('type', '')
        if query_info['intent'] == 'summary_request':
            if content_type in ['overall_summary', 'semantic_block']:
                bonus += 0.1
        elif query_info['intent'] == 'topic_search':
            if content_type == 'semantic_block':
                bonus += 0.1
        
        # Confidence bonus
        if metadata.get('confidence', 0) > 0.8:
            bonus += 0.05
        
        return min(base_score + bonus, 1.0)  # Cap at 1.0
    
    def _format_time_range(self, start_time: float, end_time: float) -> str:
        """Format time range for display"""
        def format_seconds(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        
        return f"{format_seconds(start_time)} - {format_seconds(end_time)}"
    
    def _generate_response(self, results: List[Dict], query_info: Dict, original_query: str) -> Dict:
        """Generate structured response"""
        if not results:
            return {
                'summary': f"No results found for '{original_query}'",
                'suggestions': self._get_search_suggestions(),
                'result_count': 0
            }
        
        # Generate summary based on intent
        if query_info['intent'] == 'summary_request':
            summary = self._generate_summary_response(results)
        elif query_info['intent'] == 'speaker_search':
            speaker = query_info['entities']['speakers'][0] if query_info['entities']['speakers'] else 'the speaker'
            summary = f"Found {len(results)} segments where {speaker} spoke"
        elif query_info['intent'] == 'emotion_search':
            emotion = query_info['entities']['emotions'][0] if query_info['entities']['emotions'] else 'emotional'
            summary = f"Found {len(results)} {emotion} moments in the conversation"
        elif query_info['intent'] == 'topic_search':
            topic = query_info['entities']['topics'][0] if query_info['entities']['topics'] else 'the topic'
            summary = f"Found {len(results)} segments discussing {topic}"
        else:
            summary = f"Found {len(results)} relevant results for '{original_query}'"
        
        # Extract key highlights
        highlights = self._extract_highlights(results[:3])  # Top 3 results
        
        return {
            'summary': summary,
            'highlights': highlights,
            'result_count': len(results),
            'query_intent': query_info['intent'],
            'suggestions': self._get_related_suggestions(results, query_info)
        }
    
    def _generate_summary_response(self, results: List[Dict]) -> str:
        """Generate summary response for summary queries"""
        summary_results = [r for r in results if r['metadata'].get('type') == 'overall_summary']
        
        if summary_results:
            return summary_results[0]['document']
        
        # Fallback to semantic blocks
        block_results = [r for r in results if r['metadata'].get('type') == 'semantic_block']
        if block_results:
            summaries = [r['document'] for r in block_results[:3]]
            return ' '.join(summaries)
        
        return "Summary information not available for this content."
    
    def _extract_highlights(self, results: List[Dict]) -> List[str]:
        """Extract key highlights from top results"""
        highlights = []
        
        for result in results:
            document = result['document']
            metadata = result['metadata']
            
            # Create highlight with context
            highlight = document[:200] + "..." if len(document) > 200 else document
            
            # Add metadata context
            context_parts = []
            if 'speaker' in metadata:
                context_parts.append(f"Speaker: {metadata['speaker']}")
            
            if 'time_range' in result:
                context_parts.append(f"Time: {result['time_range']}")
            
            if 'emotion_label' in metadata:
                context_parts.append(f"Emotion: {metadata['emotion_label']}")
            
            if context_parts:
                highlight += f" [{', '.join(context_parts)}]"
            
            highlights.append(highlight)
        
        return highlights
    
    def _get_search_suggestions(self) -> List[str]:
        """Get general search suggestions"""
        return [
            "Try searching for specific topics: 'about technology'",
            "Search by speaker: 'what did John say'",
            "Find emotional moments: 'happy moments'",
            "Request summaries: 'summarize the main points'",
            "Search by time: 'at 5:30'"
        ]
    
    def _get_related_suggestions(self, results: List[Dict], query_info: Dict) -> List[str]:
        """Get suggestions based on current results"""
        suggestions = []
        
        # Extract speakers from results
        speakers = set()
        emotions = set()
        topics = set()
        
        for result in results:
            metadata = result['metadata']
            
            if 'speaker' in metadata:
                speakers.add(metadata['speaker'])
            
            if 'emotion_label' in metadata:
                emotions.add(metadata['emotion_label'])
            
            if 'key_topics' in metadata:
                topics.update(metadata['key_topics'].split(','))
        
        # Generate suggestions
        if speakers:
            speaker_list = list(speakers)[:3]
            suggestions.append(f"Search by speakers: {', '.join(speaker_list)}")
        
        if emotions:
            emotion_list = list(emotions)[:3]
            suggestions.append(f"Find {'/'.join(emotion_list)} moments")
        
        if topics:
            topic_list = [t.strip() for t in list(topics)[:3] if t.strip()]
            if topic_list:
                suggestions.append(f"Explore topics: {', '.join(topic_list)}")
        
        return suggestions
    
    def search_similar_content(self, result_id: str, n_results: int = 5) -> List[Dict]:
        """Find content similar to a specific result"""
        # This would require storing result IDs and implementing similarity search
        # For now, return empty list
        logger.warning("Similar content search not yet implemented")
        return []
    
    def get_conversation_context(self, result: Dict, context_window: int = 2) -> List[Dict]:
        """Get conversation context around a specific result"""
        metadata = result['metadata']
        
        if 'start_time' in metadata and 'end_time' in metadata:
            start_time = metadata['start_time']
            end_time = metadata['end_time']
            
            # Expand time window
            context_start = max(0, start_time - context_window * 30)  # 30 seconds per window
            context_end = end_time + context_window * 30
            
            # Search for content in this timeframe
            context_results = self.vector_db.search_by_timeframe(
                context_start, context_end, n_results=10
            )
            
            # Sort by time
            context_results.sort(key=lambda x: x['metadata'].get('start_time', 0))
            
            return context_results
        
        return []
    
    def get_session_overview(self, session_id: str) -> Dict:
        """Get overview of a specific session"""
        return self.vector_db.get_session_summary(session_id)
    
    def get_popular_topics(self, session_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get most discussed topics"""
        # Search for semantic blocks (which contain topic information)
        search_params = {
            'query': '',
            'n_results': 1000,
            'content_types': ['semantic_block']
        }
        
        if session_id:
            search_params['filters'] = {'session_id': session_id}
        
        results = self.vector_db.search(**search_params)
        
        # Count topics
        topic_counts = {}
        for result in results:
            metadata = result['metadata']
            if 'key_topics' in metadata:
                topics = metadata['key_topics'].split(',')
                for topic in topics:
                    topic = topic.strip()
                    if topic:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort and format
        popular_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [{'topic': topic, 'count': count} for topic, count in popular_topics]


def create_query_processor(db_path: str = "rag_system/vector_db") -> QueryProcessor:
    """
    Convenience function to create a query processor
    
    Args:
        db_path: Path to vector database
        
    Returns:
        QueryProcessor instance
    """
    vector_db = VectorDatabase(db_path=db_path)
    return QueryProcessor(vector_db=vector_db)
