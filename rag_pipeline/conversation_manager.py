import os
import json
import time
from typing import List, Dict, Any, Optional

class ConversationManager:
    """
    Manages conversation history, user preferences, feedback, and session metadata.
    Supports context-aware responses, follow-up suggestions, and adaptive learning.
    """
    def __init__(self, storage_dir: str = "conversation_history"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.current_session: List[Dict[str, Any]] = []
        self.session_metadata: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.feedback_due = 3  # Start with feedback every 3 queries
        self.last_feedback_at = 0

    def start_new_session(self, episode_id: str, user_id: str = "default") -> str:
        """
        Start a new conversation session for an episode/user.
        Returns session_id.
        """
        session_id = f"{episode_id}_{user_id}_{int(time.time())}"
        self.session_metadata = {
            'session_id': session_id,
            'episode_id': episode_id,
            'user_id': user_id,
            'start_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'interaction_count': 0,
            'topics_discussed': [],
            'user_interests': [],
            'feedback': []
        }
        self.current_session = []
        self.last_feedback_at = 0
        return session_id

    def add_interaction(self, user_query: str, ai_response: str, sources: List[Dict[str, Any]], query_metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new interaction to the current session.
        """
        interaction = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'interaction_id': len(self.current_session) + 1,
            'user_query': user_query,
            'ai_response': ai_response,
            'sources_used': sources,
            'query_metadata': query_metadata or {},
            'topics': self._extract_topics(user_query)
        }
        self.current_session.append(interaction)
        self.session_metadata['interaction_count'] += 1
        self._update_topics(interaction['topics'])

    def get_conversation_context(self) -> str:
        """
        Returns a string summary of recent conversation for context-aware responses.
        """
        if not self.current_session:
            return ""
        lines = []
        for h in self.current_session[-5:]:
            lines.append(f"User: {h['user_query']}\nAI: {h['ai_response']}")
        return "\n".join(lines)

    def get_user_interests(self) -> List[str]:
        """
        Returns a list of user interests/topics based on session history.
        """
        return self.session_metadata.get('user_interests', [])

    def save_session(self):
        """
        Save the current session and metadata to disk.
        """
        if not self.session_metadata.get('session_id'):
            return
        session_file = os.path.join(self.storage_dir, f"session_{self.session_metadata['session_id']}.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.session_metadata,
                'interactions': self.current_session
            }, f, indent=2)

    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session by session_id.
        Returns True if successful.
        """
        session_file = os.path.join(self.storage_dir, f"session_{session_id}.json")
        if not os.path.exists(session_file):
            return False
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.session_metadata = data['metadata']
            self.current_session = data['interactions']
            return True
        except Exception:
            return False

    def feedback_due_now(self) -> bool:
        """
        Returns True if it's time to request feedback from the user.
        """
        count = self.session_metadata.get('interaction_count', 0)
        return (count - self.last_feedback_at) >= self.feedback_due

    def record_feedback(self, feedback: Dict[str, Any]):
        """
        Record user feedback and adapt feedback timing.
        """
        self.session_metadata.setdefault('feedback', []).append(feedback)
        self.last_feedback_at = self.session_metadata.get('interaction_count', 0)
        # Adapt feedback interval (simulate learning): randomize between 3-8
        import random
        self.feedback_due = random.randint(3, 8)

    def _extract_topics(self, text: str) -> List[str]:
        """
        Simple keyword extraction from user query.
        """
        words = [w.strip('.,!?') for w in text.lower().split()]
        exclude = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'to', 'of'}
        topics = [w for w in words if len(w) > 3 and w not in exclude]
        return topics[:5]

    def _update_topics(self, new_topics: List[str]):
        """
        Update session-level topic tracking and user interests.
        """
        topics = self.session_metadata.get('topics_discussed', [])
        for t in new_topics:
            if t not in topics:
                topics.append(t)
        self.session_metadata['topics_discussed'] = topics
        # Update user interests (top 5 most frequent topics)
        all_topics = topics
        from collections import Counter
        most_common = [t for t, _ in Counter(all_topics).most_common(5)]
        self.session_metadata['user_interests'] = most_common

    def suggest_followup_questions(self, current_response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """
        Generate intelligent follow-up questions based on current context and sources.
        """
        suggestions = []
        for seg in sources[:2]:
            if seg.get('block_key_points'):
                for kp in seg['block_key_points'][:2]:
                    suggestions.append(f"Can you elaborate on: {kp}?")
            if seg.get('block_summary'):
                suggestions.append(f"What more can you tell me about: {seg['block_summary'][:60]}...")
        if not suggestions:
            suggestions = [
                "What are the main themes discussed in this episode?",
                "Can you extract key quotes from the conversation?",
                "What insights can you provide about the speakers?"
            ]
        return suggestions[:3] 