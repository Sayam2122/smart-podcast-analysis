# /podcast_rag_project/conversation_manager.py

import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

class ConversationManager:
    """
    Manages conversation history, context tracking, and user interaction patterns
    for enhanced continuity and personalized responses.
    """
    
    def __init__(self, storage_dir: str = "conversation_history"):
        self.storage_dir = storage_dir
        self.current_session = []
        self.session_metadata = {}
        self.user_preferences = {}
        self.conversation_patterns = {}
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing user data if available
        self._load_user_data()
    
    def start_new_session(self, episode_id: str, user_id: str = "default") -> str:
        """
        Start a new conversation session for an episode.
        
        Args:
            episode_id: ID of the podcast episode
            user_id: User identifier for personalization
            
        Returns:
            Session ID for tracking
        """
        session_id = f"{episode_id}_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_metadata = {
            'session_id': session_id,
            'episode_id': episode_id,
            'user_id': user_id,
            'start_time': datetime.now().isoformat(),
            'interaction_count': 0,
            'topics_discussed': [],
            'user_interests': [],
            'response_quality_feedback': []
        }
        
        self.current_session = []
        logging.info(f"Started new conversation session: {session_id}")
        return session_id
    
    def add_interaction(self, user_query: str, ai_response: str, sources: List[Dict[str, Any]], 
                       query_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new interaction to the current session.
        
        Args:
            user_query: User's question/input
            ai_response: AI's response
            sources: Sources used in the response
            query_metadata: Additional metadata about the query
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'interaction_id': len(self.current_session) + 1,
            'user_query': user_query,
            'ai_response': ai_response,
            'sources_used': sources,
            'query_metadata': query_metadata or {},
            'extracted_topics': self._extract_topics(user_query),
            'response_type': self._classify_response_type(ai_response),
            'contextual_relevance': self._assess_contextual_relevance(user_query)
        }
        
        self.current_session.append(interaction)
        self.session_metadata['interaction_count'] += 1
        
        # Update session topics
        self._update_session_topics(interaction['extracted_topics'])
        
        logging.info(f"Added interaction {interaction['interaction_id']} to session")
    
    def get_conversation_context(self, max_interactions: int = 5) -> str:
        """
        Get formatted conversation history for context.
        
        Args:
            max_interactions: Maximum number of recent interactions to include
            
        Returns:
            Formatted conversation history string
        """
        if not self.current_session:
            return "No previous conversation history."
        
        recent_interactions = self.current_session[-max_interactions:]
        context_parts = []
        
        for interaction in recent_interactions:
            context_parts.append(
                f"User: {interaction['user_query']}\n"
                f"AI: {interaction['ai_response'][:200]}...\n"
                f"Topics: {', '.join(interaction['extracted_topics'])}\n"
            )
        
        return "\n--- Conversation History ---\n" + "\n".join(context_parts)
    
    def get_user_interests(self) -> List[str]:
        """Get identified user interests based on conversation history."""
        if not self.current_session:
            return []
        
        all_topics = []
        for interaction in self.current_session:
            all_topics.extend(interaction['extracted_topics'])
        
        # Count topic frequency
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Return topics mentioned more than once, sorted by frequency
        frequent_topics = [topic for topic, count in topic_counts.items() if count > 1]
        return sorted(frequent_topics, key=lambda x: topic_counts[x], reverse=True)
    
    def suggest_followup_questions(self, current_response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """
        Generate intelligent follow-up questions based on current context.
        
        Args:
            current_response: The AI's current response
            sources: Sources used in current response
            
        Returns:
            List of suggested follow-up questions
        """
        user_interests = self.get_user_interests()
        recent_topics = self.session_metadata.get('topics_discussed', [])
        
        suggestions = []
        
        # Topic-based suggestions
        if sources:
            for source in sources[:2]:  # Use top 2 sources
                suggestions.extend(self._generate_topic_questions(source, user_interests))
        
        # Conversation flow suggestions
        if len(self.current_session) > 1:
            last_interaction = self.current_session[-1]
            suggestions.extend(self._generate_flow_questions(last_interaction))
        
        # Interest-based suggestions
        for interest in user_interests[:3]:  # Top 3 interests
            suggestions.append(f"What else does this episode reveal about {interest}?")
        
        return list(set(suggestions))[:5]  # Remove duplicates, return top 5
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in the current conversation session.
        
        Returns:
            Dictionary with conversation analysis
        """
        if not self.current_session:
            return {}
        
        analysis = {
            'total_interactions': len(self.current_session),
            'session_duration_minutes': self._calculate_session_duration(),
            'avg_query_length': self._calculate_avg_query_length(),
            'topic_evolution': self._analyze_topic_evolution(),
            'question_types': self._analyze_question_types(),
            'engagement_level': self._assess_engagement_level(),
            'preferred_response_style': self._identify_response_preferences()
        }
        
        return analysis
    
    def save_session(self) -> None:
        """Save current session to persistent storage."""
        if not self.current_session:
            return
        
        session_file = os.path.join(
            self.storage_dir, 
            f"session_{self.session_metadata['session_id']}.json"
        )
        
        session_data = {
            'metadata': self.session_metadata,
            'interactions': self.current_session,
            'analysis': self.analyze_conversation_patterns()
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved session to {session_file}")
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            Success status
        """
        session_file = os.path.join(self.storage_dir, f"session_{session_id}.json")
        
        if not os.path.exists(session_file):
            logging.warning(f"Session file not found: {session_file}")
            return False
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.session_metadata = session_data['metadata']
            self.current_session = session_data['interactions']
            
            logging.info(f"Loaded session: {session_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics/keywords from user query."""
        # Simple keyword extraction (could be enhanced with NLP)
        text_lower = text.lower()
        
        # Common question words to exclude
        exclude_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        
        words = [word.strip('.,!?') for word in text_lower.split()]
        topics = [word for word in words if len(word) > 3 and word not in exclude_words]
        
        return topics[:5]  # Return top 5 potential topics
    
    def _classify_response_type(self, response: str) -> str:
        """Classify the type of AI response."""
        response_lower = response.lower()
        
        if '?' in response:
            return 'question'
        elif any(word in response_lower for word in ['quote', 'said', 'mentioned']):
            return 'quote_based'
        elif any(word in response_lower for word in ['summary', 'overall', 'main points']):
            return 'summary'
        elif any(word in response_lower for word in ['analysis', 'insight', 'suggests']):
            return 'analytical'
        else:
            return 'informational'
    
    def _assess_contextual_relevance(self, query: str) -> float:
        """Assess how relevant the query is to recent conversation context."""
        if len(self.current_session) < 2:
            return 1.0  # First question is always relevant
        
        # Simple relevance based on topic overlap
        query_topics = set(self._extract_topics(query))
        recent_topics = set()
        
        for interaction in self.current_session[-3:]:  # Last 3 interactions
            recent_topics.update(interaction['extracted_topics'])
        
        if not recent_topics:
            return 0.5
        
        overlap = len(query_topics.intersection(recent_topics))
        return min(overlap / len(query_topics) if query_topics else 0, 1.0)
    
    def _update_session_topics(self, new_topics: List[str]) -> None:
        """Update session-level topic tracking."""
        topics_discussed = self.session_metadata.get('topics_discussed', [])
        topics_discussed.extend(new_topics)
        
        # Keep unique topics, maintain order
        unique_topics = []
        for topic in topics_discussed:
            if topic not in unique_topics:
                unique_topics.append(topic)
        
        self.session_metadata['topics_discussed'] = unique_topics
    
    def _generate_topic_questions(self, source: Dict[str, Any], user_interests: List[str]) -> List[str]:
        """Generate questions based on source content and user interests."""
        questions = []
        
        # Extract key information from source
        text = source.get('text', '')
        speaker = source.get('speaker', '')
        emotion = source.get('text_emotion', {}).get('emotion', '')
        
        # Generic source-based questions
        if speaker:
            questions.append(f"What else does {speaker} discuss in this episode?")
        
        if emotion and emotion != 'neutral':
            questions.append(f"What triggered the {emotion} tone in this part?")
        
        # Interest-based questions
        for interest in user_interests[:2]:
            if interest.lower() in text.lower():
                questions.append(f"How does this relate to other mentions of {interest}?")
        
        return questions
    
    def _generate_flow_questions(self, last_interaction: Dict[str, Any]) -> List[str]:
        """Generate questions based on conversation flow."""
        questions = []
        
        response_type = last_interaction['response_type']
        
        if response_type == 'quote_based':
            questions.append("What's the broader context around this quote?")
            questions.append("Are there similar insights elsewhere in the episode?")
        elif response_type == 'summary':
            questions.append("Can you dive deeper into any of these points?")
            questions.append("What specific examples support these ideas?")
        elif response_type == 'analytical':
            questions.append("What evidence supports this analysis?")
            questions.append("How does this compare to other episodes?")
        
        return questions
    
    def _calculate_session_duration(self) -> float:
        """Calculate session duration in minutes."""
        if not self.current_session:
            return 0
        
        start_time = datetime.fromisoformat(self.session_metadata['start_time'])
        last_interaction = datetime.fromisoformat(self.current_session[-1]['timestamp'])
        
        duration = (last_interaction - start_time).total_seconds() / 60
        return round(duration, 2)
    
    def _calculate_avg_query_length(self) -> float:
        """Calculate average length of user queries."""
        if not self.current_session:
            return 0
        
        total_length = sum(len(interaction['user_query'].split()) 
                          for interaction in self.current_session)
        return round(total_length / len(self.current_session), 2)
    
    def _analyze_topic_evolution(self) -> List[str]:
        """Analyze how topics evolved during the session."""
        if len(self.current_session) < 2:
            return []
        
        topic_sequence = []
        for interaction in self.current_session:
            if interaction['extracted_topics']:
                topic_sequence.append(interaction['extracted_topics'][0])  # Primary topic
        
        return topic_sequence
    
    def _analyze_question_types(self) -> Dict[str, int]:
        """Analyze types of questions asked."""
        question_types = {}
        
        for interaction in self.current_session:
            query = interaction['user_query'].lower()
            
            if query.startswith('what'):
                q_type = 'what'
            elif query.startswith('how'):
                q_type = 'how'
            elif query.startswith('why'):
                q_type = 'why'
            elif query.startswith('when'):
                q_type = 'when'
            elif query.startswith('where'):
                q_type = 'where'
            else:
                q_type = 'other'
            
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return question_types
    
    def _assess_engagement_level(self) -> str:
        """Assess user engagement level based on interaction patterns."""
        if len(self.current_session) < 2:
            return 'low'
        
        # Factors for engagement
        interaction_count = len(self.current_session)
        avg_query_length = self._calculate_avg_query_length()
        contextual_queries = sum(1 for interaction in self.current_session 
                               if interaction['contextual_relevance'] > 0.7)
        
        engagement_score = 0
        
        if interaction_count > 5:
            engagement_score += 2
        elif interaction_count > 2:
            engagement_score += 1
        
        if avg_query_length > 8:
            engagement_score += 2
        elif avg_query_length > 5:
            engagement_score += 1
        
        if contextual_queries > len(self.current_session) * 0.6:
            engagement_score += 2
        
        if engagement_score >= 4:
            return 'high'
        elif engagement_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _identify_response_preferences(self) -> str:
        """Identify user's preferred response style."""
        if not self.current_session:
            return 'unknown'
        
        # Analyze query patterns to infer preferences
        detail_seeking = sum(1 for interaction in self.current_session 
                           if any(word in interaction['user_query'].lower() 
                                 for word in ['detail', 'explain', 'elaborate', 'more']))
        
        quote_seeking = sum(1 for interaction in self.current_session 
                          if any(word in interaction['user_query'].lower() 
                                for word in ['quote', 'said', 'mentioned', 'exact']))
        
        summary_seeking = sum(1 for interaction in self.current_session 
                            if any(word in interaction['user_query'].lower() 
                                  for word in ['summary', 'overview', 'main points']))
        
        total_interactions = len(self.current_session)
        
        if detail_seeking / total_interactions > 0.5:
            return 'detailed'
        elif quote_seeking / total_interactions > 0.3:
            return 'quote_focused'
        elif summary_seeking / total_interactions > 0.3:
            return 'summary_focused'
        else:
            return 'balanced'
    
    def _load_user_data(self) -> None:
        """Load existing user preferences and patterns."""
        user_data_file = os.path.join(self.storage_dir, "user_data.json")
        
        if os.path.exists(user_data_file):
            try:
                with open(user_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_preferences = data.get('preferences', {})
                    self.conversation_patterns = data.get('patterns', {})
            except Exception as e:
                logging.warning(f"Could not load user data: {e}")
    
    def save_user_data(self) -> None:
        """Save user preferences and patterns."""
        user_data_file = os.path.join(self.storage_dir, "user_data.json")
        
        data = {
            'preferences': self.user_preferences,
            'patterns': self.conversation_patterns,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(user_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Could not save user data: {e}")
