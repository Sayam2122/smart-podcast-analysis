"""
Feedback System for Audio RAG

Advanced feedback collection and learning system with
variable timing, pattern analysis, and continuous improvement.
"""

import json
import sqlite3
import threading
import random
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    SATISFACTION = "satisfaction"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    USEFULNESS = "usefulness"
    SUGGESTION = "suggestion"


class FeedbackTiming(Enum):
    """When to collect feedback."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    SESSION_END = "session_end"
    PERIODIC = "periodic"
    TRIGGERED = "triggered"


@dataclass
class FeedbackPrompt:
    """Configuration for feedback prompts."""
    feedback_type: FeedbackType
    timing: FeedbackTiming
    prompt_text: str
    response_options: List[str]
    probability: float  # Probability of showing this prompt
    conditions: Dict[str, Any]  # Conditions for showing prompt


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    id: str
    timestamp: datetime
    user_id: str
    conversation_id: str
    interaction_id: Optional[str]
    feedback_type: FeedbackType
    rating: Optional[float]
    text_feedback: Optional[str]
    metadata: Dict[str, Any]


class FeedbackSystem:
    """
    Advanced feedback collection and learning system.
    
    Features:
    - Variable timing feedback collection
    - Pattern-based learning
    - Adaptive prompting
    - Performance impact analysis
    """
    
    def __init__(self, storage_path: str = "data/feedback.db"):
        """
        Initialize feedback system.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Feedback prompts configuration
        self._initialize_prompts()
        
        # Feedback collection state
        self._pending_feedback: Dict[str, List[Dict]] = {}  # user_id -> pending feedback
        self._feedback_schedule: Dict[str, datetime] = {}  # user_id -> next feedback time
        
        # Learning patterns
        self._learned_patterns: Dict[str, Any] = {}
        
        # Configuration
        self.config = {
            'min_feedback_interval': 300,  # 5 minutes between feedback requests
            'max_daily_feedback': 5,      # Max feedback requests per day per user
            'delayed_feedback_hours': [1, 4, 24],  # Hours for delayed feedback
            'satisfaction_threshold': 0.7,  # Threshold for good satisfaction
            'feedback_fatigue_limit': 3,   # Consecutive low-rating limit before reducing frequency
        }
        
        # Statistics
        self.stats = {
            'total_feedback': 0,
            'feedback_by_type': {ft.value: 0 for ft in FeedbackType},
            'avg_satisfaction': 0,
            'feedback_response_rate': 0,
            'improvement_metrics': {}
        }
        
        self._update_statistics()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    interaction_id TEXT,
                    feedback_type TEXT NOT NULL,
                    rating REAL,
                    text_feedback TEXT,
                    metadata TEXT  -- JSON string
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_prompts_shown (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    prompt_type TEXT NOT NULL,
                    shown_at TEXT NOT NULL,
                    responded BOOLEAN DEFAULT 0,
                    response_time_seconds INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback_preferences (
                    user_id TEXT PRIMARY KEY,
                    frequency_preference TEXT DEFAULT 'normal',
                    preferred_timing TEXT DEFAULT 'delayed',
                    feedback_types_enabled TEXT,  -- JSON array
                    last_updated TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,  -- JSON string
                    confidence REAL DEFAULT 0.5,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_entries(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON feedback_entries(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_entries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_user ON feedback_prompts_shown(user_id)")
            
            conn.commit()
    
    def _initialize_prompts(self):
        """Initialize feedback prompt configurations."""
        self.feedback_prompts = {
            FeedbackType.SATISFACTION: [
                FeedbackPrompt(
                    feedback_type=FeedbackType.SATISFACTION,
                    timing=FeedbackTiming.IMMEDIATE,
                    prompt_text="How satisfied are you with this response?",
                    response_options=["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"],
                    probability=0.3,
                    conditions={"min_response_length": 50}
                ),
                FeedbackPrompt(
                    feedback_type=FeedbackType.SATISFACTION,
                    timing=FeedbackTiming.DELAYED,
                    prompt_text="Looking back at our earlier conversation, how helpful was my response?",
                    response_options=["Very helpful", "Helpful", "Somewhat helpful", "Not very helpful", "Not helpful at all"],
                    probability=0.2,
                    conditions={"delay_hours": 4}
                )
            ],
            
            FeedbackType.RELEVANCE: [
                FeedbackPrompt(
                    feedback_type=FeedbackType.RELEVANCE,
                    timing=FeedbackTiming.IMMEDIATE,
                    prompt_text="How relevant was this response to your question?",
                    response_options=["Very relevant", "Relevant", "Somewhat relevant", "Not very relevant", "Not relevant"],
                    probability=0.2,
                    conditions={"query_complexity": "high"}
                )
            ],
            
            FeedbackType.COMPLETENESS: [
                FeedbackPrompt(
                    feedback_type=FeedbackType.COMPLETENESS,
                    timing=FeedbackTiming.IMMEDIATE,
                    prompt_text="Did this response fully answer your question?",
                    response_options=["Completely", "Mostly", "Partially", "Barely", "Not at all"],
                    probability=0.15,
                    conditions={"intent_type": "analytical"}
                )
            ],
            
            FeedbackType.ACCURACY: [
                FeedbackPrompt(
                    feedback_type=FeedbackType.ACCURACY,
                    timing=FeedbackTiming.DELAYED,
                    prompt_text="Was the information in my response accurate?",
                    response_options=["Very accurate", "Accurate", "Mostly accurate", "Somewhat inaccurate", "Very inaccurate"],
                    probability=0.1,
                    conditions={"delay_hours": 1, "contains_facts": True}
                )
            ],
            
            FeedbackType.SUGGESTION: [
                FeedbackPrompt(
                    feedback_type=FeedbackType.SUGGESTION,
                    timing=FeedbackTiming.SESSION_END,
                    prompt_text="How can I improve my responses? (Optional)",
                    response_options=[],  # Free text
                    probability=0.1,
                    conditions={"session_length": 5}
                )
            ]
        }
    
    def should_request_feedback(self, 
                              user_id: str,
                              conversation_id: str,
                              interaction_context: Dict[str, Any]) -> Optional[FeedbackPrompt]:
        """
        Determine if feedback should be requested based on context and timing.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            interaction_context: Context of the current interaction
            
        Returns:
            FeedbackPrompt if feedback should be requested, None otherwise
        """
        with self._lock:
            # Check if user has feedback fatigue
            if self._has_feedback_fatigue(user_id):
                return None
            
            # Check daily feedback limit
            if self._exceeded_daily_limit(user_id):
                return None
            
            # Check minimum interval
            if not self._minimum_interval_passed(user_id):
                return None
            
            # Get user preferences
            user_prefs = self._get_user_feedback_preferences(user_id)
            
            # Find applicable prompts
            applicable_prompts = []
            
            for feedback_type, prompts in self.feedback_prompts.items():
                if feedback_type.value not in user_prefs.get('feedback_types_enabled', [ft.value for ft in FeedbackType]):
                    continue
                
                for prompt in prompts:
                    if self._check_prompt_conditions(prompt, interaction_context, user_prefs):
                        applicable_prompts.append(prompt)
            
            # Select prompt based on probability
            if applicable_prompts:
                selected_prompt = self._select_prompt_by_probability(applicable_prompts)
                if selected_prompt:
                    self._record_prompt_shown(user_id, selected_prompt.feedback_type.value)
                    return selected_prompt
            
            return None
    
    def collect_feedback(self,
                        user_id: str,
                        conversation_id: str,
                        feedback_type: FeedbackType,
                        rating: Optional[float] = None,
                        text_feedback: Optional[str] = None,
                        interaction_id: Optional[str] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """
        Collect and store user feedback.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            feedback_type: Type of feedback
            rating: Numerical rating (0-1 scale)
            text_feedback: Free-text feedback
            interaction_id: Specific interaction being rated
            metadata: Additional metadata
            
        Returns:
            Feedback entry ID
        """
        feedback_id = f"{user_id}_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        feedback_entry = FeedbackEntry(
            id=feedback_id,
            timestamp=datetime.now(),
            user_id=user_id,
            conversation_id=conversation_id,
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            rating=rating,
            text_feedback=text_feedback,
            metadata=metadata or {}
        )
        
        # Store feedback
        self._store_feedback_entry(feedback_entry)
        
        # Update user feedback schedule
        self._update_feedback_schedule(user_id)
        
        # Learn from feedback
        self._learn_from_feedback(feedback_entry)
        
        # Update statistics
        self._update_statistics()
        
        logger.info(f"Collected {feedback_type.value} feedback from user {user_id}")
        
        return feedback_id
    
    def schedule_delayed_feedback(self,
                                user_id: str,
                                conversation_id: str,
                                interaction_id: str,
                                delay_hours: int = 4):
        """Schedule feedback collection for later."""
        schedule_time = datetime.now() + timedelta(hours=delay_hours)
        
        feedback_item = {
            'user_id': user_id,
            'conversation_id': conversation_id,
            'interaction_id': interaction_id,
            'scheduled_for': schedule_time,
            'feedback_type': FeedbackType.SATISFACTION.value
        }
        
        if user_id not in self._pending_feedback:
            self._pending_feedback[user_id] = []
        
        self._pending_feedback[user_id].append(feedback_item)
        
        logger.info(f"Scheduled delayed feedback for user {user_id} in {delay_hours} hours")
    
    def check_pending_feedback(self, user_id: str) -> List[Dict[str, Any]]:
        """Check for pending feedback requests for a user."""
        if user_id not in self._pending_feedback:
            return []
        
        now = datetime.now()
        ready_feedback = []
        remaining_feedback = []
        
        for feedback_item in self._pending_feedback[user_id]:
            if feedback_item['scheduled_for'] <= now:
                ready_feedback.append(feedback_item)
            else:
                remaining_feedback.append(feedback_item)
        
        # Update pending feedback list
        self._pending_feedback[user_id] = remaining_feedback
        
        return ready_feedback
    
    def get_feedback_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get comprehensive feedback analytics.
        
        Args:
            user_id: Optional user filter
            
        Returns:
            Analytics dictionary
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                analytics = {}
                
                # Base filter
                base_filter = ""
                params = []
                if user_id:
                    base_filter = "WHERE user_id = ?"
                    params = [user_id]
                
                # Total feedback count
                cursor = conn.execute(f"SELECT COUNT(*) FROM feedback_entries {base_filter}", params)
                analytics['total_feedback'] = cursor.fetchone()[0]
                
                # Feedback by type
                cursor = conn.execute(f"""
                    SELECT feedback_type, COUNT(*) 
                    FROM feedback_entries {base_filter}
                    GROUP BY feedback_type
                """, params)
                analytics['feedback_by_type'] = dict(cursor.fetchall())
                
                # Average ratings by type
                cursor = conn.execute(f"""
                    SELECT feedback_type, AVG(rating), COUNT(*)
                    FROM feedback_entries 
                    {base_filter} AND rating IS NOT NULL
                    GROUP BY feedback_type
                """, params)
                
                ratings_by_type = {}
                for feedback_type, avg_rating, count in cursor.fetchall():
                    ratings_by_type[feedback_type] = {
                        'average': avg_rating,
                        'count': count
                    }
                analytics['ratings_by_type'] = ratings_by_type
                
                # Overall satisfaction trend
                cursor = conn.execute(f"""
                    SELECT DATE(timestamp) as date, AVG(rating) as avg_rating
                    FROM feedback_entries 
                    {base_filter} AND feedback_type = 'satisfaction' AND rating IS NOT NULL
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 30
                """, params)
                
                satisfaction_trend = []
                for date, avg_rating in cursor.fetchall():
                    satisfaction_trend.append({
                        'date': date,
                        'average_satisfaction': avg_rating
                    })
                analytics['satisfaction_trend'] = satisfaction_trend
                
                # Response rate analysis
                cursor = conn.execute(f"""
                    SELECT 
                        COUNT(*) as prompts_shown,
                        SUM(CASE WHEN responded = 1 THEN 1 ELSE 0 END) as responses
                    FROM feedback_prompts_shown {base_filter}
                """, params)
                
                prompts_shown, responses = cursor.fetchone()
                if prompts_shown > 0:
                    analytics['response_rate'] = responses / prompts_shown
                else:
                    analytics['response_rate'] = 0
                
                # Text feedback themes (simple keyword analysis)
                cursor = conn.execute(f"""
                    SELECT text_feedback 
                    FROM feedback_entries 
                    {base_filter} AND text_feedback IS NOT NULL
                """, params)
                
                text_feedbacks = [row[0] for row in cursor.fetchall()]
                analytics['text_feedback_themes'] = self._analyze_text_feedback_themes(text_feedbacks)
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get feedback analytics: {e}")
            return {}
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get AI-generated suggestions for system improvement based on feedback patterns."""
        analytics = self.get_feedback_analytics()
        suggestions = []
        
        # Low satisfaction areas
        if 'ratings_by_type' in analytics:
            for feedback_type, data in analytics['ratings_by_type'].items():
                if data['average'] < 0.6 and data['count'] >= 5:
                    suggestions.append({
                        'priority': 'high',
                        'area': feedback_type,
                        'issue': f"Low average rating ({data['average']:.2f})",
                        'suggestion': f"Focus on improving {feedback_type} aspects of responses",
                        'evidence': f"Based on {data['count']} feedback entries"
                    })
        
        # Response relevance issues
        if analytics.get('response_rate', 0) < 0.3:
            suggestions.append({
                'priority': 'medium',
                'area': 'user_engagement',
                'issue': f"Low feedback response rate ({analytics.get('response_rate', 0):.2f})",
                'suggestion': "Consider adjusting feedback timing and frequency",
                'evidence': "Users may be experiencing feedback fatigue"
            })
        
        # Text feedback analysis
        themes = analytics.get('text_feedback_themes', {})
        negative_themes = {k: v for k, v in themes.items() if any(neg in k.lower() for neg in ['slow', 'wrong', 'bad', 'poor', 'error'])}
        
        for theme, count in negative_themes.items():
            if count >= 3:
                suggestions.append({
                    'priority': 'medium',
                    'area': 'response_quality',
                    'issue': f"Recurring negative feedback theme: '{theme}'",
                    'suggestion': f"Investigate and address {theme} issues",
                    'evidence': f"Mentioned in {count} feedback entries"
                })
        
        return suggestions
    
    def update_user_feedback_preferences(self,
                                       user_id: str,
                                       frequency: str = None,
                                       timing: str = None,
                                       enabled_types: List[str] = None):
        """Update user's feedback preferences."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Get current preferences
                cursor = conn.execute("""
                    SELECT frequency_preference, preferred_timing, feedback_types_enabled
                    FROM user_feedback_preferences
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    current_freq, current_timing, current_types = row
                    current_types_list = json.loads(current_types) if current_types else [ft.value for ft in FeedbackType]
                else:
                    current_freq = 'normal'
                    current_timing = 'delayed'
                    current_types_list = [ft.value for ft in FeedbackType]
                
                # Update with new values
                new_freq = frequency or current_freq
                new_timing = timing or current_timing
                new_types = enabled_types or current_types_list
                
                # Save updated preferences
                conn.execute("""
                    INSERT OR REPLACE INTO user_feedback_preferences
                    (user_id, frequency_preference, preferred_timing, feedback_types_enabled, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    new_freq,
                    new_timing,
                    json.dumps(new_types),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                logger.info(f"Updated feedback preferences for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to update user feedback preferences: {e}")
    
    def export_feedback_data(self, user_id: str = None) -> Dict[str, Any]:
        """Export feedback data for analysis."""
        analytics = self.get_feedback_analytics(user_id)
        
        # Get raw feedback entries
        try:
            with sqlite3.connect(self.storage_path) as conn:
                base_filter = ""
                params = []
                if user_id:
                    base_filter = "WHERE user_id = ?"
                    params = [user_id]
                
                cursor = conn.execute(f"""
                    SELECT timestamp, user_id, conversation_id, feedback_type, 
                           rating, text_feedback, metadata
                    FROM feedback_entries {base_filter}
                    ORDER BY timestamp DESC
                """, params)
                
                raw_entries = []
                for row in cursor.fetchall():
                    timestamp, uid, conv_id, fb_type, rating, text, metadata_str = row
                    
                    entry = {
                        'timestamp': timestamp,
                        'user_id': uid,
                        'conversation_id': conv_id,
                        'feedback_type': fb_type,
                        'rating': rating,
                        'text_feedback': text,
                        'metadata': json.loads(metadata_str) if metadata_str else {}
                    }
                    raw_entries.append(entry)
                
                return {
                    'analytics': analytics,
                    'raw_entries': raw_entries,
                    'export_timestamp': datetime.now().isoformat(),
                    'total_entries': len(raw_entries)
                }
                
        except Exception as e:
            logger.error(f"Failed to export feedback data: {e}")
            return {'error': str(e)}
    
    def _has_feedback_fatigue(self, user_id: str) -> bool:
        """Check if user has feedback fatigue."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Check recent low satisfaction ratings
                cursor = conn.execute("""
                    SELECT rating FROM feedback_entries
                    WHERE user_id = ? AND feedback_type = 'satisfaction' 
                    AND rating IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, self.config['feedback_fatigue_limit']))
                
                recent_ratings = [row[0] for row in cursor.fetchall()]
                
                if len(recent_ratings) >= self.config['feedback_fatigue_limit']:
                    avg_rating = sum(recent_ratings) / len(recent_ratings)
                    return avg_rating < 0.3  # Very low satisfaction
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to check feedback fatigue: {e}")
            return False
    
    def _exceeded_daily_limit(self, user_id: str) -> bool:
        """Check if user has exceeded daily feedback limit."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                today = datetime.now().date().isoformat()
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM feedback_prompts_shown
                    WHERE user_id = ? AND DATE(shown_at) = ?
                """, (user_id, today))
                
                daily_count = cursor.fetchone()[0]
                return daily_count >= self.config['max_daily_feedback']
                
        except Exception as e:
            logger.error(f"Failed to check daily limit: {e}")
            return False
    
    def _minimum_interval_passed(self, user_id: str) -> bool:
        """Check if minimum interval has passed since last feedback request."""
        if user_id not in self._feedback_schedule:
            return True
        
        last_feedback_time = self._feedback_schedule[user_id]
        min_interval = timedelta(seconds=self.config['min_feedback_interval'])
        
        return datetime.now() - last_feedback_time >= min_interval
    
    def _get_user_feedback_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's feedback preferences."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT frequency_preference, preferred_timing, feedback_types_enabled
                    FROM user_feedback_preferences
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    freq, timing, types_str = row
                    return {
                        'frequency_preference': freq,
                        'preferred_timing': timing,
                        'feedback_types_enabled': json.loads(types_str) if types_str else [ft.value for ft in FeedbackType]
                    }
                else:
                    # Default preferences
                    return {
                        'frequency_preference': 'normal',
                        'preferred_timing': 'delayed',
                        'feedback_types_enabled': [ft.value for ft in FeedbackType]
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get user feedback preferences: {e}")
            return {}
    
    def _check_prompt_conditions(self, 
                               prompt: FeedbackPrompt, 
                               context: Dict[str, Any],
                               user_prefs: Dict[str, Any]) -> bool:
        """Check if prompt conditions are met."""
        conditions = prompt.conditions
        
        # Check timing preference
        if user_prefs.get('preferred_timing') and prompt.timing.value != user_prefs['preferred_timing']:
            # Allow some flexibility
            if random.random() > 0.3:
                return False
        
        # Check specific conditions
        for condition, value in conditions.items():
            if condition == 'min_response_length':
                response_length = len(context.get('response', ''))
                if response_length < value:
                    return False
            
            elif condition == 'query_complexity':
                # Simple heuristic for query complexity
                query = context.get('query', '')
                complexity = 'high' if len(query.split()) > 10 else 'low'
                if complexity != value:
                    return False
            
            elif condition == 'intent_type':
                if context.get('intent_type') != value:
                    return False
            
            elif condition == 'contains_facts':
                # Check if response contains factual information
                response = context.get('response', '').lower()
                fact_indicators = ['according to', 'research shows', 'study found', 'data indicates']
                has_facts = any(indicator in response for indicator in fact_indicators)
                if has_facts != value:
                    return False
            
            elif condition == 'session_length':
                session_length = context.get('interaction_count', 0)
                if session_length < value:
                    return False
        
        return True
    
    def _select_prompt_by_probability(self, prompts: List[FeedbackPrompt]) -> Optional[FeedbackPrompt]:
        """Select a prompt based on probability weights."""
        if not prompts:
            return None
        
        # Simple probability-based selection
        for prompt in prompts:
            if random.random() < prompt.probability:
                return prompt
        
        return None
    
    def _record_prompt_shown(self, user_id: str, prompt_type: str):
        """Record that a feedback prompt was shown to user."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO feedback_prompts_shown
                    (user_id, prompt_type, shown_at)
                    VALUES (?, ?, ?)
                """, (user_id, prompt_type, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record prompt shown: {e}")
    
    def _store_feedback_entry(self, feedback: FeedbackEntry):
        """Store feedback entry in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO feedback_entries
                    (id, timestamp, user_id, conversation_id, interaction_id,
                     feedback_type, rating, text_feedback, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id,
                    feedback.timestamp.isoformat(),
                    feedback.user_id,
                    feedback.conversation_id,
                    feedback.interaction_id,
                    feedback.feedback_type.value,
                    feedback.rating,
                    feedback.text_feedback,
                    json.dumps(feedback.metadata)
                ))
                
                # Mark prompt as responded if applicable
                conn.execute("""
                    UPDATE feedback_prompts_shown
                    SET responded = 1, response_time_seconds = ?
                    WHERE user_id = ? AND prompt_type = ? AND responded = 0
                    ORDER BY shown_at DESC
                    LIMIT 1
                """, (
                    0,  # Could calculate actual response time
                    feedback.user_id,
                    feedback.feedback_type.value
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store feedback entry: {e}")
    
    def _update_feedback_schedule(self, user_id: str):
        """Update user's feedback schedule."""
        self._feedback_schedule[user_id] = datetime.now()
    
    def _learn_from_feedback(self, feedback: FeedbackEntry):
        """Learn patterns from feedback for system improvement."""
        # This is where machine learning could be implemented
        # For now, we'll do simple pattern detection
        
        pattern_key = f"{feedback.feedback_type.value}_{feedback.user_id}"
        
        if pattern_key not in self._learned_patterns:
            self._learned_patterns[pattern_key] = {
                'feedback_count': 0,
                'rating_sum': 0,
                'common_themes': {},
                'improvement_areas': []
            }
        
        pattern = self._learned_patterns[pattern_key]
        pattern['feedback_count'] += 1
        
        if feedback.rating is not None:
            pattern['rating_sum'] += feedback.rating
        
        # Analyze text feedback for themes
        if feedback.text_feedback:
            themes = self._extract_themes_from_text(feedback.text_feedback)
            for theme in themes:
                if theme in pattern['common_themes']:
                    pattern['common_themes'][theme] += 1
                else:
                    pattern['common_themes'][theme] = 1
    
    def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract themes from text feedback using simple keyword analysis."""
        import re
        
        # Simple keyword-based theme extraction
        themes = []
        
        text_lower = text.lower()
        
        # Positive themes
        positive_keywords = ['good', 'great', 'helpful', 'useful', 'accurate', 'fast', 'clear']
        for keyword in positive_keywords:
            if keyword in text_lower:
                themes.append(f"positive_{keyword}")
        
        # Negative themes
        negative_keywords = ['bad', 'wrong', 'slow', 'unclear', 'unhelpful', 'inaccurate', 'confusing']
        for keyword in negative_keywords:
            if keyword in text_lower:
                themes.append(f"negative_{keyword}")
        
        # Feature-related themes
        feature_keywords = ['search', 'response', 'timing', 'relevance', 'length', 'detail']
        for keyword in feature_keywords:
            if keyword in text_lower:
                themes.append(f"feature_{keyword}")
        
        return themes
    
    def _analyze_text_feedback_themes(self, text_feedbacks: List[str]) -> Dict[str, int]:
        """Analyze themes across multiple text feedback entries."""
        theme_counts = {}
        
        for text in text_feedbacks:
            themes = self._extract_themes_from_text(text)
            for theme in themes:
                if theme in theme_counts:
                    theme_counts[theme] += 1
                else:
                    theme_counts[theme] = 1
        
        # Sort by frequency
        sorted_themes = dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_themes
    
    def _update_statistics(self):
        """Update feedback system statistics."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Total feedback
                cursor = conn.execute("SELECT COUNT(*) FROM feedback_entries")
                self.stats['total_feedback'] = cursor.fetchone()[0]
                
                # Feedback by type
                cursor = conn.execute("""
                    SELECT feedback_type, COUNT(*) 
                    FROM feedback_entries 
                    GROUP BY feedback_type
                """)
                for feedback_type, count in cursor.fetchall():
                    self.stats['feedback_by_type'][feedback_type] = count
                
                # Average satisfaction
                cursor = conn.execute("""
                    SELECT AVG(rating) 
                    FROM feedback_entries 
                    WHERE feedback_type = 'satisfaction' AND rating IS NOT NULL
                """)
                result = cursor.fetchone()[0]
                self.stats['avg_satisfaction'] = result if result else 0
                
                # Response rate
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN responded = 1 THEN 1 ELSE 0 END) as responded
                    FROM feedback_prompts_shown
                """)
                total, responded = cursor.fetchone()
                if total > 0:
                    self.stats['feedback_response_rate'] = responded / total
                else:
                    self.stats['feedback_response_rate'] = 0
                
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current feedback system statistics."""
        self._update_statistics()
        return dict(self.stats)
