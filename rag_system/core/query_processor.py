"""
Smart Query Processor for Audio RAG System

Intent-aware query processing with multi-dimensional routing,
conversation context integration, and advanced filtering capabilities.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    CONTENT = "content"
    SPEAKER = "speaker"
    EMOTION = "emotion"
    TEMPORAL = "temporal"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


@dataclass
class QueryIntent:
    """Structured representation of query intent."""
    query_type: QueryType
    primary_focus: str
    entities: Dict[str, Any]
    filters: Dict[str, Any]
    confidence: float
    requires_context: bool = False


class SmartQueryProcessor:
    """
    Advanced query processor with intent recognition and multi-modal routing.
    
    Features:
    - Intent classification and entity extraction
    - Multi-dimensional query routing
    - Conversation context integration
    - Temporal and speaker-aware processing
    """
    
    def __init__(self, vector_db, conversation_memory=None):
        """
        Initialize the Smart Query Processor.
        
        Args:
            vector_db: SmartVectorDB instance
            conversation_memory: ConversationMemory instance (optional)
        """
        self.vector_db = vector_db
        self.conversation_memory = conversation_memory
        
        # Initialize query patterns and mappings
        self._initialize_patterns()
        self._initialize_entity_extractors()
        self._initialize_response_templates()
        
        # Query statistics
        self.query_stats = {
            'total_queries': 0,
            'by_type': {qt.value: 0 for qt in QueryType},
            'successful_queries': 0,
            'avg_processing_time': 0
        }
    
    def _initialize_patterns(self):
        """Initialize regex patterns for intent recognition."""
        self.intent_patterns = {
            QueryType.SPEAKER: [
                r'\b(?:who|speaker|person)\s+(?:said|spoke|mentioned|talked about|discussed)',
                r'\b(?:what did|tell me what)\s+\w+\s+(?:say|said|talk|spoke)',
                r'\bspeaker\s*\d+',
                r'\b(?:host|guest|participant|moderator)\b',
                r'\b(?:john|jane|mike|sarah|mary|bob|alice|charlie|david|emma)\b'
            ],
            
            QueryType.EMOTION: [
                r'\b(?:emotion|feeling|mood|sentiment|tone)\b',
                r'\b(?:happy|sad|angry|excited|calm|frustrated|positive|negative|neutral)\b',
                r'\b(?:emotional|emotionally)\s+\w+',
                r'\b(?:how did .+ feel|what was the mood|emotional response)\b',
                r'\b(?:excited|enthusiastic|passionate|concerned|worried|upset)\b'
            ],
            
            QueryType.TEMPORAL: [
                r'\b(?:when|at what time|around|during|between|from|after|before)\b',
                r'\b\d{1,2}:\d{2}\b',  # Time format
                r'\b\d+\s*(?:minutes?|mins?|seconds?|secs?|hours?)\b',
                r'\b(?:beginning|start|end|middle|later|earlier|first|last)\b',
                r'\b(?:timeline|chronology|sequence|order)\b'
            ],
            
            QueryType.ANALYTICAL: [
                r'\b(?:analyze|analysis|pattern|trend|comparison|statistics)\b',
                r'\b(?:summary|summarize|overview|key points|main ideas)\b',
                r'\b(?:insights|conclusions|themes|topics|categories)\b',
                r'\b(?:most|least|average|total|percentage|distribution)\b',
                r'\b(?:compare|contrast|difference|similarity)\b'
            ],
            
            QueryType.CONVERSATIONAL: [
                r'\b(?:tell me more|continue|elaborate|expand)\b',
                r'\b(?:what else|anything else|more details|additional)\b',
                r'\b(?:follow up|related to|similar|like that)\b',
                r'\b(?:that|this|it|they)\b.*(?:mentioned|discussed|said)'
            ]
        }
        
        # Content type preferences by query type
        self.content_preferences = {
            QueryType.CONTENT: ['segment', 'block', 'episode'],
            QueryType.SPEAKER: ['speaker', 'segment'],
            QueryType.EMOTION: ['emotion', 'segment'],
            QueryType.TEMPORAL: ['segment', 'block'],
            QueryType.ANALYTICAL: ['episode', 'block', 'speaker'],
            QueryType.CONVERSATIONAL: ['segment', 'block']
        }
    
    def _initialize_entity_extractors(self):
        """Initialize entity extraction patterns."""
        self.entity_patterns = {
            'speakers': [
                r'\bspeaker\s*(\d+)\b',
                r'\b(host|guest|moderator|participant)\b',
                r'\b(john|jane|mike|sarah|mary|bob|alice|charlie|david|emma)\b'
            ],
            
            'emotions': [
                r'\b(happy|happiness|joy|joyful|excited|excitement|enthusiastic)\b',
                r'\b(sad|sadness|depressed|melancholy|down|disappointed)\b',
                r'\b(angry|anger|mad|frustrated|irritated|annoyed)\b',
                r'\b(calm|peaceful|relaxed|composed|neutral)\b',
                r'\b(surprised|surprised|amazed|shocked|astonished)\b',
                r'\b(fearful|afraid|scared|anxious|worried|nervous)\b'
            ],
            
            'time_references': [
                r'\b(\d{1,2}):(\d{2})\b',  # MM:SS format
                r'\b(\d+)\s*(?:minutes?|mins?)\b',
                r'\b(\d+)\s*(?:seconds?|secs?)\b',
                r'\b(beginning|start|end|middle|later|earlier)\b'
            ],
            
            'topics': [
                r'\babout\s+([a-zA-Z\s]+?)(?:\s+(?:and|or|but)|[.?!]|$)',
                r'\b(?:discuss|discussion|talk|talked about|regarding|concerning)\s+([a-zA-Z\s]+?)(?:\s+(?:and|or|but)|[.?!]|$)',
                r'\btopic\s+(?:of\s+)?([a-zA-Z\s]+?)(?:\s+(?:and|or|but)|[.?!]|$)'
            ]
        }
    
    def _initialize_response_templates(self):
        """Initialize response templates for different query types."""
        self.response_templates = {
            QueryType.SPEAKER: {
                'intro': "Based on the speaker analysis:",
                'no_results': "I couldn't find any specific speaker information for that query.",
                'context_fields': ['speaker', 'speaker_confidence', 'speaking_ratio']
            },
            
            QueryType.EMOTION: {
                'intro': "Here's what I found about the emotional content:",
                'no_results': "I couldn't find any emotional content matching that description.",
                'context_fields': ['emotion', 'emotion_confidence', 'emotion_intensity']
            },
            
            QueryType.TEMPORAL: {
                'intro': "Here's what happened at that time:",
                'no_results': "I couldn't find any content for that time period.",
                'context_fields': ['start_time', 'end_time', 'duration']
            },
            
            QueryType.ANALYTICAL: {
                'intro': "Here's the analytical summary:",
                'no_results': "I couldn't generate an analysis for that request.",
                'context_fields': ['importance', 'confidence', 'themes']
            },
            
            QueryType.CONTENT: {
                'intro': "Here's what I found:",
                'no_results': "I couldn't find any relevant content for that query.",
                'context_fields': ['content_type', 'confidence', 'similarity_score']
            }
        }
    
    def process_query(self, 
                     query: str,
                     session_ids: List[str] = None,
                     conversation_id: str = None,
                     user_id: str = None) -> Dict[str, Any]:
        """
        Process a natural language query with full intent analysis.
        
        Args:
            query: Natural language query
            session_ids: Specific sessions to search (None for all loaded)
            conversation_id: Conversation context identifier
            user_id: User identifier for personalization
            
        Returns:
            Comprehensive query response with results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Update statistics
            self.query_stats['total_queries'] += 1
            
            # Analyze query intent
            intent = self._analyze_intent(query)
            
            # Extract entities
            entities = self._extract_entities(query)
            
            # Add conversation context if available
            if self.conversation_memory and conversation_id:
                context = self.conversation_memory.get_context(conversation_id)
                intent, entities = self._integrate_context(intent, entities, context)
            
            # Build search parameters
            search_params = self._build_search_params(intent, entities, session_ids)
            
            # Execute search
            raw_results = self.vector_db.search(**search_params)
            
            # Post-process and rank results
            processed_results = self._process_results(raw_results, intent, entities)
            
            # Generate response
            response = self._generate_response(processed_results, intent, query)
            
            # Store in conversation memory
            if self.conversation_memory and conversation_id:
                self.conversation_memory.store_interaction(
                    conversation_id, query, response, user_id
                )
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(intent.query_type, processing_time, True)
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(None, processing_time, False)
            
            return {
                'query': query,
                'success': False,
                'error': str(e),
                'response': "I encountered an error processing your query. Please try rephrasing it.",
                'results': [],
                'metadata': {
                    'processing_time': processing_time,
                    'intent': None,
                    'entities': {}
                }
            }
    
    def _analyze_intent(self, query: str) -> QueryIntent:
        """Analyze query to determine intent and primary focus."""
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        
        for query_type, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    score += len(matches)
                    matched_patterns.append(pattern)
            
            if score > 0:
                type_scores[query_type] = {
                    'score': score,
                    'patterns': matched_patterns
                }
        
        # Determine primary intent
        if type_scores:
            primary_type = max(type_scores.keys(), key=lambda x: type_scores[x]['score'])
            confidence = min(0.9, 0.5 + type_scores[primary_type]['score'] * 0.1)
        else:
            # Default to content search
            primary_type = QueryType.CONTENT
            confidence = 0.3
        
        # Determine if context is required
        requires_context = self._requires_conversation_context(query_lower)
        
        return QueryIntent(
            query_type=primary_type,
            primary_focus=self._extract_primary_focus(query),
            entities={},  # Will be filled by entity extraction
            filters={},   # Will be built from entities
            confidence=confidence,
            requires_context=requires_context
        )
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract named entities and structured information from query."""
        entities = {
            'speakers': [],
            'emotions': [],
            'time_references': {},
            'topics': [],
            'numbers': [],
            'filters': {}
        }
        
        query_lower = query.lower()
        
        # Extract speakers
        for pattern in self.entity_patterns['speakers']:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities['speakers'].extend(matches)
        
        # Extract emotions
        for pattern in self.entity_patterns['emotions']:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities['emotions'].extend(matches)
        
        # Extract time references
        time_refs = {}
        for pattern in self.entity_patterns['time_references']:
            matches = re.findall(pattern, query_lower)
            if matches:
                if ':' in pattern:  # Time format MM:SS
                    for match in matches:
                        if isinstance(match, tuple):
                            minutes, seconds = match
                            time_refs['timestamp'] = int(minutes) * 60 + int(seconds)
                elif 'minutes' in pattern or 'mins' in pattern:
                    for match in matches:
                        time_refs['minutes'] = int(match) if match.isdigit() else 0
                elif 'seconds' in pattern or 'secs' in pattern:
                    for match in matches:
                        time_refs['seconds'] = int(match) if match.isdigit() else 0
                else:
                    time_refs['relative'] = matches[0] if matches else None
        
        entities['time_references'] = time_refs
        
        # Extract topics
        for pattern in self.entity_patterns['topics']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['topics'].extend([match.strip() for match in matches if match.strip()])
        
        # Extract numbers for potential filtering
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities['numbers'] = [float(n) for n in numbers]
        
        # Build filters from entities
        entities['filters'] = self._build_entity_filters(entities)
        
        return entities
    
    def _build_entity_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters from extracted entities."""
        filters = {}
        
        # Speaker filters
        if entities['speakers']:
            # Use the first speaker mentioned
            speaker = entities['speakers'][0]
            if speaker.isdigit():
                filters['speaker'] = f"Speaker_{speaker.zfill(2)}"
            else:
                filters['speaker'] = speaker.title()
        
        # Emotion filters
        if entities['emotions']:
            filters['emotion'] = entities['emotions'][0]
        
        # Time filters
        time_refs = entities['time_references']
        if 'timestamp' in time_refs:
            # Search around the specified timestamp (±30 seconds)
            timestamp = time_refs['timestamp']
            filters['time_range'] = (max(0, timestamp - 30), timestamp + 30)
        elif 'minutes' in time_refs:
            # Convert minutes to seconds
            start_time = time_refs['minutes'] * 60
            filters['time_range'] = (start_time, start_time + 60)
        elif 'seconds' in time_refs:
            # Search around the specified second
            filters['time_range'] = (time_refs['seconds'], time_refs['seconds'] + 10)
        
        # Confidence filters
        if 'high quality' in entities.get('query_text', '').lower():
            filters['min_confidence'] = 0.8
        elif 'low confidence' in entities.get('query_text', '').lower():
            filters['max_confidence'] = 0.5
        
        return filters
    
    def _integrate_context(self, 
                          intent: QueryIntent, 
                          entities: Dict[str, Any], 
                          context: Dict[str, Any]) -> Tuple[QueryIntent, Dict[str, Any]]:
        """Integrate conversation context into current query processing."""
        if not context:
            return intent, entities
        
        # Get recent context
        recent_queries = context.get('recent_queries', [])
        current_topic = context.get('current_topic')
        user_preferences = context.get('user_preferences', {})
        
        # Enhance entities with context
        if current_topic and not entities['topics']:
            entities['topics'] = [current_topic]
        
        # Resolve pronouns and references
        entities = self._resolve_references(entities, recent_queries)
        
        # Apply user preferences
        if user_preferences:
            entities['filters'].update(self._apply_user_preferences(user_preferences))
        
        # Adjust intent confidence based on context
        if intent.requires_context and recent_queries:
            intent.confidence = min(0.95, intent.confidence + 0.2)
        
        return intent, entities
    
    def _resolve_references(self, 
                          entities: Dict[str, Any], 
                          recent_queries: List[Dict]) -> Dict[str, Any]:
        """Resolve pronouns and references using conversation history."""
        if not recent_queries:
            return entities
        
        # Look for speaker references in recent context
        for query_data in recent_queries[-3:]:  # Check last 3 queries
            query_entities = query_data.get('entities', {})
            
            if query_entities.get('speakers') and not entities['speakers']:
                entities['speakers'] = query_entities['speakers']
            
            if query_entities.get('topics') and not entities['topics']:
                entities['topics'] = query_entities['topics']
        
        return entities
    
    def _build_search_params(self, 
                           intent: QueryIntent, 
                           entities: Dict[str, Any], 
                           session_ids: List[str] = None) -> Dict[str, Any]:
        """Build search parameters for vector database query."""
        # Determine content types based on intent
        content_types = self.content_preferences.get(
            intent.query_type, 
            ['segment', 'block', 'episode']
        )
        
        # Build search parameters
        params = {
            'query': intent.primary_focus,
            'content_types': content_types,
            'session_ids': session_ids,
            'filters': entities['filters'],
            'limit': self._determine_result_limit(intent)
        }
        
        return params
    
    def _determine_result_limit(self, intent: QueryIntent) -> int:
        """Determine appropriate result limit based on query intent."""
        limits = {
            QueryType.CONTENT: 10,
            QueryType.SPEAKER: 15,  # Might need more for speaker analysis
            QueryType.EMOTION: 12,
            QueryType.TEMPORAL: 8,   # Usually more focused
            QueryType.ANALYTICAL: 20,  # Need more for analysis
            QueryType.CONVERSATIONAL: 5   # More focused follow-ups
        }
        
        return limits.get(intent.query_type, 10)
    
    def _process_results(self, 
                        raw_results: List[Dict], 
                        intent: QueryIntent, 
                        entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process and rank search results based on intent."""
        if not raw_results:
            return []
        
        processed_results = []
        
        for result in raw_results:
            # Calculate relevance score
            relevance_score = self._calculate_relevance(result, intent, entities)
            
            # Add processed metadata
            processed_result = {
                **result,
                'relevance_score': relevance_score,
                'intent_match': self._check_intent_match(result, intent),
                'entity_match': self._check_entity_match(result, entities),
                'context_relevance': self._calculate_context_relevance(result, intent)
            }
            
            processed_results.append(processed_result)
        
        # Sort by combined score
        processed_results.sort(
            key=lambda x: (x['relevance_score'], x['similarity_score']), 
            reverse=True
        )
        
        return processed_results
    
    def _calculate_relevance(self, 
                           result: Dict, 
                           intent: QueryIntent, 
                           entities: Dict[str, Any]) -> float:
        """Calculate relevance score based on intent and entities."""
        score = result.get('similarity_score', 0)
        
        metadata = result.get('metadata', {})
        
        # Boost for intent-specific matches
        if intent.query_type == QueryType.SPEAKER:
            if metadata.get('speaker') in entities.get('speakers', []):
                score += 0.3
        
        elif intent.query_type == QueryType.EMOTION:
            if metadata.get('emotion') in entities.get('emotions', []):
                score += 0.3
        
        elif intent.query_type == QueryType.TEMPORAL:
            time_range = entities['filters'].get('time_range')
            if time_range:
                start_time = metadata.get('start_time', 0)
                if time_range[0] <= start_time <= time_range[1]:
                    score += 0.3
        
        # Boost for high confidence content
        confidence = metadata.get('confidence', 0)
        if confidence > 0.8:
            score += 0.1
        elif confidence < 0.5:
            score -= 0.1
        
        # Boost for important content
        importance = metadata.get('importance', '')
        if importance == 'high':
            score += 0.15
        elif importance == 'low':
            score -= 0.05
        
        return min(1.0, max(0.0, score))
    
    def _check_intent_match(self, result: Dict, intent: QueryIntent) -> bool:
        """Check if result matches the query intent."""
        content_type = result.get('content_type', '')
        preferred_types = self.content_preferences.get(intent.query_type, [])
        
        return content_type in preferred_types
    
    def _check_entity_match(self, result: Dict, entities: Dict[str, Any]) -> Dict[str, bool]:
        """Check which entities match in the result."""
        metadata = result.get('metadata', {})
        matches = {}
        
        # Speaker match
        result_speaker = metadata.get('speaker', '')
        matches['speaker'] = any(
            speaker.lower() in result_speaker.lower() 
            for speaker in entities.get('speakers', [])
        )
        
        # Emotion match
        result_emotion = metadata.get('emotion', '')
        matches['emotion'] = result_emotion in entities.get('emotions', [])
        
        # Topic match (check in content)
        content = result.get('content', '').lower()
        matches['topic'] = any(
            topic.lower() in content 
            for topic in entities.get('topics', [])
        )
        
        return matches
    
    def _calculate_context_relevance(self, result: Dict, intent: QueryIntent) -> float:
        """Calculate how relevant the result is to the current context."""
        # This can be enhanced with conversation memory
        base_relevance = 0.5
        
        # Boost recent content for conversational queries
        if intent.query_type == QueryType.CONVERSATIONAL:
            base_relevance += 0.2
        
        # Boost comprehensive content for analytical queries
        if intent.query_type == QueryType.ANALYTICAL:
            content_length = len(result.get('content', ''))
            if content_length > 200:  # Longer content for analysis
                base_relevance += 0.1
        
        return base_relevance
    
    def _generate_response(self, 
                          results: List[Dict], 
                          intent: QueryIntent, 
                          original_query: str) -> Dict[str, Any]:
        """Generate comprehensive response with results and metadata."""
        template = self.response_templates.get(intent.query_type, {})
        
        if not results:
            response_text = template.get('no_results', "I couldn't find relevant information for your query.")
        else:
            # Generate response based on query type
            if intent.query_type == QueryType.ANALYTICAL:
                response_text = self._generate_analytical_response(results)
            elif intent.query_type == QueryType.SPEAKER:
                response_text = self._generate_speaker_response(results)
            elif intent.query_type == QueryType.EMOTION:
                response_text = self._generate_emotion_response(results)
            elif intent.query_type == QueryType.TEMPORAL:
                response_text = self._generate_temporal_response(results)
            else:
                response_text = self._generate_content_response(results)
        
        return {
            'query': original_query,
            'success': True,
            'response': response_text,
            'results': results[:10],  # Limit displayed results
            'metadata': {
                'intent': {
                    'type': intent.query_type.value,
                    'focus': intent.primary_focus,
                    'confidence': intent.confidence
                },
                'result_count': len(results),
                'processing_time': 0,  # Will be filled by caller
                'content_types': list(set(r.get('content_type', '') for r in results)),
                'sessions_searched': list(set(r.get('metadata', {}).get('session_id', '') for r in results))
            }
        }
    
    def _generate_analytical_response(self, results: List[Dict]) -> str:
        """Generate analytical summary response."""
        if not results:
            return "No data available for analysis."
        
        # Analyze patterns in results
        speakers = set()
        emotions = set()
        topics = set()
        timeframes = []
        
        for result in results[:10]:  # Analyze top 10 results
            metadata = result.get('metadata', {})
            speakers.add(metadata.get('speaker', 'Unknown'))
            emotions.add(metadata.get('emotion', 'neutral'))
            
            # Extract topics from content
            content = result.get('content', '')
            topics.update(self._extract_quick_topics(content))
            
            start_time = metadata.get('start_time', 0)
            if start_time:
                timeframes.append(start_time)
        
        # Generate summary
        response_parts = ["Based on my analysis:"]
        
        if len(speakers) > 1:
            response_parts.append(f"• {len(speakers)} speakers were involved: {', '.join(list(speakers)[:3])}")
        
        if emotions:
            dominant_emotions = list(emotions)[:3]
            response_parts.append(f"• Dominant emotions: {', '.join(dominant_emotions)}")
        
        if topics:
            main_topics = list(topics)[:5]
            response_parts.append(f"• Key topics discussed: {', '.join(main_topics)}")
        
        if timeframes:
            duration = max(timeframes) - min(timeframes) if len(timeframes) > 1 else 0
            response_parts.append(f"• Content spans approximately {duration/60:.1f} minutes")
        
        return "\n".join(response_parts)
    
    def _generate_speaker_response(self, results: List[Dict]) -> str:
        """Generate speaker-focused response."""
        if not results:
            return "No speaker information found."
        
        # Group by speaker
        speaker_content = {}
        for result in results[:5]:
            speaker = result.get('metadata', {}).get('speaker', 'Unknown')
            if speaker not in speaker_content:
                speaker_content[speaker] = []
            speaker_content[speaker].append(result.get('content', '')[:200])
        
        response_parts = ["Here's what I found about the speakers:"]
        
        for speaker, content_list in speaker_content.items():
            combined_content = " ".join(content_list)
            response_parts.append(f"\n**{speaker}:**")
            response_parts.append(f"  {combined_content[:300]}...")
        
        return "\n".join(response_parts)
    
    def _generate_emotion_response(self, results: List[Dict]) -> str:
        """Generate emotion-focused response."""
        if not results:
            return "No emotional content found."
        
        # Group by emotion
        emotion_content = {}
        for result in results[:5]:
            emotion = result.get('metadata', {}).get('emotion', 'neutral')
            confidence = result.get('metadata', {}).get('emotion_confidence', 0)
            
            if emotion not in emotion_content:
                emotion_content[emotion] = []
            
            emotion_content[emotion].append({
                'content': result.get('content', '')[:200],
                'confidence': confidence,
                'speaker': result.get('metadata', {}).get('speaker', 'Unknown')
            })
        
        response_parts = ["Here's the emotional content I found:"]
        
        for emotion, content_list in emotion_content.items():
            avg_confidence = sum(c['confidence'] for c in content_list) / len(content_list)
            response_parts.append(f"\n**{emotion.title()} (confidence: {avg_confidence:.2f}):**")
            
            for item in content_list[:2]:  # Show top 2 per emotion
                response_parts.append(f"  [{item['speaker']}] {item['content']}...")
        
        return "\n".join(response_parts)
    
    def _generate_temporal_response(self, results: List[Dict]) -> str:
        """Generate time-focused response."""
        if not results:
            return "No content found for that time period."
        
        # Sort by time
        timed_results = sorted(
            results[:5],
            key=lambda x: x.get('metadata', {}).get('start_time', 0)
        )
        
        response_parts = ["Here's what happened at that time:"]
        
        for result in timed_results:
            metadata = result.get('metadata', {})
            start_time = metadata.get('start_time', 0)
            speaker = metadata.get('speaker', 'Unknown')
            content = result.get('content', '')[:200]
            
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            
            response_parts.append(f"\n**{minutes:02d}:{seconds:02d} - {speaker}:**")
            response_parts.append(f"  {content}...")
        
        return "\n".join(response_parts)
    
    def _generate_content_response(self, results: List[Dict]) -> str:
        """Generate general content response."""
        if not results:
            return "No relevant content found."
        
        # Show top results with context
        response_parts = ["Here's what I found:"]
        
        for i, result in enumerate(results[:3], 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            speaker = metadata.get('speaker', '')
            similarity = result.get('similarity_score', 0)
            
            response_parts.append(f"\n**Result {i} (relevance: {similarity:.2f}):**")
            if speaker:
                response_parts.append(f"  Speaker: {speaker}")
            response_parts.append(f"  {content[:300]}...")
        
        return "\n".join(response_parts)
    
    def _extract_quick_topics(self, content: str) -> List[str]:
        """Quick topic extraction from content."""
        # Simple keyword extraction
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        topics = [word for word in words if word not in stop_words]
        
        # Return most frequent unique topics
        from collections import Counter
        topic_counts = Counter(topics)
        return [topic for topic, count in topic_counts.most_common(5)]
    
    def _extract_primary_focus(self, query: str) -> str:
        """Extract the primary focus/subject of the query."""
        # Remove question words and common phrases
        focus = re.sub(r'\b(?:what|when|where|who|how|why|tell me|show me|find|search)\b', '', query, flags=re.IGNORECASE)
        focus = re.sub(r'\b(?:about|regarding|concerning)\b', '', focus, flags=re.IGNORECASE)
        
        return focus.strip()
    
    def _requires_conversation_context(self, query: str) -> bool:
        """Determine if query requires conversation context."""
        context_indicators = [
            r'\b(?:that|this|it|they)\b',
            r'\b(?:tell me more|continue|elaborate)\b',
            r'\b(?:what else|anything else|more details)\b',
            r'\b(?:follow up|related to|similar)\b'
        ]
        
        return any(re.search(pattern, query) for pattern in context_indicators)
    
    def _apply_user_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to search filters."""
        filters = {}
        
        # Preferred speakers
        if 'preferred_speakers' in preferences:
            # This could boost certain speakers in ranking
            pass
        
        # Content type preferences
        if 'preferred_content_types' in preferences:
            # This could adjust content type ordering
            pass
        
        # Quality thresholds
        if 'min_confidence' in preferences:
            filters['min_confidence'] = preferences['min_confidence']
        
        return filters
    
    def _update_statistics(self, 
                          query_type: Optional[QueryType], 
                          processing_time: float, 
                          success: bool):
        """Update query processing statistics."""
        if success:
            self.query_stats['successful_queries'] += 1
        
        if query_type:
            self.query_stats['by_type'][query_type.value] += 1
        
        # Update running average of processing time
        total_queries = self.query_stats['total_queries']
        current_avg = self.query_stats['avg_processing_time']
        new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        self.query_stats['avg_processing_time'] = new_avg
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query processing statistics."""
        stats = dict(self.query_stats)
        
        # Calculate success rate
        if stats['total_queries'] > 0:
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']
        else:
            stats['success_rate'] = 0
        
        # Add distribution percentages
        total = stats['total_queries']
        if total > 0:
            for query_type in stats['by_type']:
                stats['by_type'][query_type] = {
                    'count': stats['by_type'][query_type],
                    'percentage': (stats['by_type'][query_type] / total) * 100
                }
        
        return stats
    
    def reset_statistics(self):
        """Reset all query statistics."""
        self.query_stats = {
            'total_queries': 0,
            'by_type': {qt.value: 0 for qt in QueryType},
            'successful_queries': 0,
            'avg_processing_time': 0
        }
