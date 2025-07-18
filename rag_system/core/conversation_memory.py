"""
Conversation Memory System for Audio RAG

Persistent conversation tracking with context management,
user preferences, and interaction history storage.
"""

import json
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """Represents a single user-system interaction."""
    timestamp: datetime
    user_id: str
    conversation_id: str
    query: str
    response: str
    intent_type: str
    entities: Dict[str, Any]
    satisfaction_score: Optional[float] = None
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ConversationContext:
    """Current conversation context and state."""
    conversation_id: str
    user_id: str
    started_at: datetime
    last_updated: datetime
    current_topic: Optional[str]
    recent_queries: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    session_focus: List[str]  # Session IDs user is focusing on
    interaction_count: int = 0


class ConversationMemory:
    """
    Manages conversation history, context, and user preferences.
    
    Features:
    - Persistent interaction storage
    - Context-aware query enhancement
    - User preference learning
    - Session-based memory management
    """
    
    def __init__(self, storage_path: str = "data/conversation_memory.db"):
        """
        Initialize conversation memory system.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Active conversations cache
        self._active_conversations: Dict[str, ConversationContext] = {}
        
        # User preferences cache
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load active conversations
        self._load_active_conversations()
        
        # Configuration
        self.max_context_queries = 10  # Max queries to keep in context
        self.context_expire_hours = 24  # Hours before context expires
        self.interaction_batch_size = 100  # Batch size for database writes
        
        # Statistics
        self.stats = {
            'total_interactions': 0,
            'active_conversations': 0,
            'unique_users': 0,
            'avg_conversation_length': 0
        }
        
        self._update_statistics()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent_type TEXT,
                    entities TEXT,  -- JSON string
                    satisfaction_score REAL,
                    feedback TEXT,
                    metadata TEXT  -- JSON string
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    current_topic TEXT,
                    recent_queries TEXT,  -- JSON string
                    user_preferences TEXT,  -- JSON string
                    session_focus TEXT,  -- JSON string
                    interaction_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,  -- JSON string
                    learned_patterns TEXT,  -- JSON string
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_conversation ON interactions(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
            
            conn.commit()
    
    def start_conversation(self, 
                          user_id: str, 
                          conversation_id: Optional[str] = None,
                          initial_context: Dict[str, Any] = None) -> str:
        """
        Start a new conversation or resume an existing one.
        
        Args:
            user_id: Unique user identifier
            conversation_id: Optional conversation ID (generated if None)
            initial_context: Initial conversation context
            
        Returns:
            Conversation ID
        """
        with self._lock:
            if conversation_id is None:
                conversation_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Check if conversation already exists
            if conversation_id in self._active_conversations:
                context = self._active_conversations[conversation_id]
                context.last_updated = datetime.now()
                return conversation_id
            
            # Load user preferences
            user_prefs = self._load_user_preferences(user_id)
            
            # Create new conversation context
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                started_at=datetime.now(),
                last_updated=datetime.now(),
                current_topic=initial_context.get('topic') if initial_context else None,
                recent_queries=[],
                user_preferences=user_prefs,
                session_focus=initial_context.get('session_focus', []) if initial_context else []
            )
            
            # Store in active conversations
            self._active_conversations[conversation_id] = context
            
            # Persist to database
            self._save_conversation_context(context)
            
            self._update_statistics()
            
            logger.info(f"Started conversation {conversation_id} for user {user_id}")
            
            return conversation_id
    
    def store_interaction(self, 
                         conversation_id: str,
                         query: str,
                         response: Any,
                         user_id: str = None,
                         satisfaction_score: float = None,
                         feedback: str = None) -> bool:
        """
        Store a user-system interaction.
        
        Args:
            conversation_id: Conversation identifier
            query: User's query
            response: System's response (dict or string)
            user_id: User identifier (optional if conversation exists)
            satisfaction_score: User satisfaction (0-1)
            feedback: User feedback text
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                # Get or create conversation context
                if conversation_id not in self._active_conversations:
                    if user_id:
                        self.start_conversation(user_id, conversation_id)
                    else:
                        logger.error(f"Conversation {conversation_id} not found and no user_id provided")
                        return False
                
                context = self._active_conversations[conversation_id]
                
                # Extract response details
                if isinstance(response, dict):
                    response_text = response.get('response', str(response))
                    intent_type = response.get('metadata', {}).get('intent', {}).get('type', 'unknown')
                    entities = response.get('metadata', {}).get('entities', {})
                    metadata = response.get('metadata', {})
                else:
                    response_text = str(response)
                    intent_type = 'unknown'
                    entities = {}
                    metadata = {}
                
                # Create interaction
                interaction = Interaction(
                    timestamp=datetime.now(),
                    user_id=context.user_id,
                    conversation_id=conversation_id,
                    query=query,
                    response=response_text,
                    intent_type=intent_type,
                    entities=entities,
                    satisfaction_score=satisfaction_score,
                    feedback=feedback,
                    metadata=metadata
                )
                
                # Store in database
                self._save_interaction(interaction)
                
                # Update conversation context
                self._update_conversation_context(context, query, response, entities)
                
                # Update user preferences based on interaction
                self._update_user_preferences(context.user_id, interaction)
                
                logger.info(f"Stored interaction for conversation {conversation_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            return False
    
    def get_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current conversation context.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Context dictionary or None if not found
        """
        with self._lock:
            if conversation_id not in self._active_conversations:
                # Try to load from database
                context = self._load_conversation_context(conversation_id)
                if context:
                    self._active_conversations[conversation_id] = context
                else:
                    return None
            
            context = self._active_conversations[conversation_id]
            
            # Check if context is still valid
            if self._is_context_expired(context):
                self._archive_conversation(conversation_id)
                return None
            
            return {
                'conversation_id': context.conversation_id,
                'user_id': context.user_id,
                'current_topic': context.current_topic,
                'recent_queries': context.recent_queries,
                'user_preferences': context.user_preferences,
                'session_focus': context.session_focus,
                'interaction_count': context.interaction_count,
                'started_at': context.started_at.isoformat(),
                'last_updated': context.last_updated.isoformat()
            }
    
    def update_topic(self, conversation_id: str, topic: str):
        """Update the current topic for a conversation."""
        with self._lock:
            if conversation_id in self._active_conversations:
                context = self._active_conversations[conversation_id]
                context.current_topic = topic
                context.last_updated = datetime.now()
                self._save_conversation_context(context)
    
    def update_session_focus(self, conversation_id: str, session_ids: List[str]):
        """Update the session focus for a conversation."""
        with self._lock:
            if conversation_id in self._active_conversations:
                context = self._active_conversations[conversation_id]
                context.session_focus = session_ids
                context.last_updated = datetime.now()
                self._save_conversation_context(context)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences with learned patterns."""
        return self._load_user_preferences(user_id)
    
    def update_user_preference(self, 
                              user_id: str, 
                              preference_key: str, 
                              preference_value: Any):
        """Update a specific user preference."""
        with self._lock:
            prefs = self._load_user_preferences(user_id)
            prefs[preference_key] = preference_value
            self._save_user_preferences(user_id, prefs)
            
            # Update all active conversations for this user
            for context in self._active_conversations.values():
                if context.user_id == user_id:
                    context.user_preferences = prefs
    
    def get_conversation_history(self, 
                               conversation_id: str, 
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction dictionaries
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, query, response, intent_type, entities, 
                           satisfaction_score, feedback, metadata
                    FROM interactions
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (conversation_id, limit))
                
                interactions = []
                for row in cursor.fetchall():
                    timestamp, query, response, intent_type, entities_str, satisfaction, feedback, metadata_str = row
                    
                    interaction = {
                        'timestamp': timestamp,
                        'query': query,
                        'response': response,
                        'intent_type': intent_type,
                        'entities': json.loads(entities_str) if entities_str else {},
                        'satisfaction_score': satisfaction,
                        'feedback': feedback,
                        'metadata': json.loads(metadata_str) if metadata_str else {}
                    }
                    
                    interactions.append(interaction)
                
                return interactions
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def get_user_conversation_list(self, user_id: str) -> List[Dict[str, Any]]:
        """Get list of conversations for a user."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT conversation_id, started_at, last_updated, 
                           current_topic, interaction_count, is_active
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY last_updated DESC
                """, (user_id,))
                
                conversations = []
                for row in cursor.fetchall():
                    conv_id, started, updated, topic, count, active = row
                    
                    conversation = {
                        'conversation_id': conv_id,
                        'started_at': started,
                        'last_updated': updated,
                        'current_topic': topic,
                        'interaction_count': count,
                        'is_active': bool(active)
                    }
                    
                    conversations.append(conversation)
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to get user conversations: {e}")
            return []
    
    def add_feedback(self, 
                    conversation_id: str, 
                    interaction_index: int, 
                    satisfaction_score: float,
                    feedback: str = None):
        """Add feedback to a specific interaction."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Get the interaction (ordered by timestamp)
                cursor = conn.execute("""
                    SELECT id FROM interactions
                    WHERE conversation_id = ?
                    ORDER BY timestamp
                    LIMIT 1 OFFSET ?
                """, (conversation_id, interaction_index))
                
                row = cursor.fetchone()
                if row:
                    interaction_id = row[0]
                    
                    # Update with feedback
                    conn.execute("""
                        UPDATE interactions
                        SET satisfaction_score = ?, feedback = ?
                        WHERE id = ?
                    """, (satisfaction_score, feedback, interaction_id))
                    
                    conn.commit()
                    logger.info(f"Added feedback to interaction {interaction_id}")
                    return True
                else:
                    logger.warning(f"Interaction not found: {conversation_id}[{interaction_index}]")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False
    
    def cleanup_expired_conversations(self):
        """Archive expired conversations and clean up memory."""
        with self._lock:
            expired_conversations = []
            
            for conv_id, context in list(self._active_conversations.items()):
                if self._is_context_expired(context):
                    expired_conversations.append(conv_id)
            
            for conv_id in expired_conversations:
                self._archive_conversation(conv_id)
            
            logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")
    
    def get_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """Get conversation analytics and patterns."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                analytics = {}
                
                # Base query filter
                base_filter = ""
                params = []
                if user_id:
                    base_filter = "WHERE user_id = ?"
                    params = [user_id]
                
                # Total interactions
                cursor = conn.execute(f"SELECT COUNT(*) FROM interactions {base_filter}", params)
                analytics['total_interactions'] = cursor.fetchone()[0]
                
                # Intent type distribution
                cursor = conn.execute(f"""
                    SELECT intent_type, COUNT(*) 
                    FROM interactions {base_filter}
                    GROUP BY intent_type
                """, params)
                analytics['intent_distribution'] = dict(cursor.fetchall())
                
                # Average satisfaction
                cursor = conn.execute(f"""
                    SELECT AVG(satisfaction_score) 
                    FROM interactions 
                    {base_filter} AND satisfaction_score IS NOT NULL
                """, params)
                result = cursor.fetchone()[0]
                analytics['avg_satisfaction'] = result if result else 0
                
                # Conversation patterns
                cursor = conn.execute(f"""
                    SELECT AVG(interaction_count), COUNT(DISTINCT conversation_id)
                    FROM conversations {base_filter}
                """, params)
                avg_length, total_convs = cursor.fetchone()
                analytics['avg_conversation_length'] = avg_length if avg_length else 0
                analytics['total_conversations'] = total_convs
                
                # Popular topics
                cursor = conn.execute(f"""
                    SELECT current_topic, COUNT(*) 
                    FROM conversations 
                    {base_filter} AND current_topic IS NOT NULL
                    GROUP BY current_topic
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """, params)
                analytics['popular_topics'] = dict(cursor.fetchall())
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}
    
    def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Export complete conversation data."""
        context = self.get_context(conversation_id)
        history = self.get_conversation_history(conversation_id)
        
        return {
            'context': context,
            'history': history,
            'exported_at': datetime.now().isoformat()
        }
    
    def _save_interaction(self, interaction: Interaction):
        """Save interaction to database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO interactions 
                    (timestamp, user_id, conversation_id, query, response, 
                     intent_type, entities, satisfaction_score, feedback, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction.timestamp.isoformat(),
                    interaction.user_id,
                    interaction.conversation_id,
                    interaction.query,
                    interaction.response,
                    interaction.intent_type,
                    json.dumps(interaction.entities),
                    interaction.satisfaction_score,
                    interaction.feedback,
                    json.dumps(interaction.metadata) if interaction.metadata else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
    
    def _save_conversation_context(self, context: ConversationContext):
        """Save conversation context to database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO conversations
                    (conversation_id, user_id, started_at, last_updated, 
                     current_topic, recent_queries, user_preferences, 
                     session_focus, interaction_count, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.conversation_id,
                    context.user_id,
                    context.started_at.isoformat(),
                    context.last_updated.isoformat(),
                    context.current_topic,
                    json.dumps(context.recent_queries),
                    json.dumps(context.user_preferences),
                    json.dumps(context.session_focus),
                    context.interaction_count,
                    1  # is_active
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save conversation context: {e}")
    
    def _load_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation context from database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT user_id, started_at, last_updated, current_topic,
                           recent_queries, user_preferences, session_focus, 
                           interaction_count
                    FROM conversations
                    WHERE conversation_id = ? AND is_active = 1
                """, (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    user_id, started, updated, topic, queries_str, prefs_str, focus_str, count = row
                    
                    return ConversationContext(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        started_at=datetime.fromisoformat(started),
                        last_updated=datetime.fromisoformat(updated),
                        current_topic=topic,
                        recent_queries=json.loads(queries_str) if queries_str else [],
                        user_preferences=json.loads(prefs_str) if prefs_str else {},
                        session_focus=json.loads(focus_str) if focus_str else [],
                        interaction_count=count
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to load conversation context: {e}")
            return None
    
    def _load_active_conversations(self):
        """Load active conversations from database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT conversation_id FROM conversations
                    WHERE is_active = 1 
                    AND datetime(last_updated) > datetime('now', '-24 hours')
                """)
                
                for row in cursor.fetchall():
                    conv_id = row[0]
                    context = self._load_conversation_context(conv_id)
                    if context:
                        self._active_conversations[conv_id] = context
                        
        except Exception as e:
            logger.error(f"Failed to load active conversations: {e}")
    
    def _update_conversation_context(self, 
                                   context: ConversationContext,
                                   query: str,
                                   response: Any,
                                   entities: Dict[str, Any]):
        """Update conversation context with new interaction."""
        # Add to recent queries
        query_data = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'entities': entities,
            'intent_type': getattr(response, 'intent_type', 'unknown') if hasattr(response, 'intent_type') else 'unknown'
        }
        
        context.recent_queries.append(query_data)
        
        # Keep only recent queries
        if len(context.recent_queries) > self.max_context_queries:
            context.recent_queries = context.recent_queries[-self.max_context_queries:]
        
        # Update topic if we can extract one
        topics = entities.get('topics', [])
        if topics:
            context.current_topic = topics[0]
        
        # Update counters and timestamp
        context.interaction_count += 1
        context.last_updated = datetime.now()
        
        # Save updated context
        self._save_conversation_context(context)
    
    def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences from database."""
        if user_id in self._user_preferences:
            return self._user_preferences[user_id].copy()
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT preferences FROM user_preferences
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    prefs = json.loads(row[0])
                    self._user_preferences[user_id] = prefs
                    return prefs.copy()
                else:
                    # Return default preferences
                    default_prefs = {
                        'preferred_response_length': 'medium',
                        'preferred_content_types': ['segment', 'block'],
                        'min_confidence': 0.5,
                        'show_timestamps': True,
                        'show_speakers': True,
                        'show_emotions': False,
                        'max_results': 10
                    }
                    self._user_preferences[user_id] = default_prefs
                    return default_prefs.copy()
                    
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
            return {}
    
    def _save_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Save user preferences to database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences
                    (user_id, preferences, learned_patterns, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id,
                    json.dumps(preferences),
                    json.dumps({}),  # Placeholder for learned patterns
                    datetime.now().isoformat()
                ))
                conn.commit()
                
                # Update cache
                self._user_preferences[user_id] = preferences
                
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
    
    def _update_user_preferences(self, user_id: str, interaction: Interaction):
        """Update user preferences based on interaction patterns."""
        # This is where machine learning could be applied
        # For now, we'll do simple pattern detection
        
        prefs = self._load_user_preferences(user_id)
        
        # Learn from satisfaction scores
        if interaction.satisfaction_score is not None:
            if interaction.satisfaction_score > 0.8:
                # User liked this type of response
                if 'successful_intent_types' not in prefs:
                    prefs['successful_intent_types'] = {}
                
                intent_type = interaction.intent_type
                if intent_type in prefs['successful_intent_types']:
                    prefs['successful_intent_types'][intent_type] += 1
                else:
                    prefs['successful_intent_types'][intent_type] = 1
        
        # Learn content type preferences from queries
        entities = interaction.entities
        if entities.get('filters', {}).get('content_type'):
            content_type = entities['filters']['content_type']
            if 'preferred_content_types' not in prefs:
                prefs['preferred_content_types'] = []
            
            if content_type not in prefs['preferred_content_types']:
                prefs['preferred_content_types'].append(content_type)
        
        # Save updated preferences
        self._save_user_preferences(user_id, prefs)
    
    def _is_context_expired(self, context: ConversationContext) -> bool:
        """Check if conversation context has expired."""
        expiry_time = datetime.now() - timedelta(hours=self.context_expire_hours)
        return context.last_updated < expiry_time
    
    def _archive_conversation(self, conversation_id: str):
        """Archive an expired conversation."""
        try:
            # Mark as inactive in database
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    UPDATE conversations
                    SET is_active = 0
                    WHERE conversation_id = ?
                """, (conversation_id,))
                conn.commit()
            
            # Remove from active conversations
            if conversation_id in self._active_conversations:
                del self._active_conversations[conversation_id]
            
            logger.info(f"Archived conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to archive conversation: {e}")
    
    def _update_statistics(self):
        """Update internal statistics."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Total interactions
                cursor = conn.execute("SELECT COUNT(*) FROM interactions")
                self.stats['total_interactions'] = cursor.fetchone()[0]
                
                # Active conversations
                self.stats['active_conversations'] = len(self._active_conversations)
                
                # Unique users
                cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                self.stats['unique_users'] = cursor.fetchone()[0]
                
                # Average conversation length
                cursor = conn.execute("SELECT AVG(interaction_count) FROM conversations")
                result = cursor.fetchone()[0]
                self.stats['avg_conversation_length'] = result if result else 0
                
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current memory system statistics."""
        self._update_statistics()
        return dict(self.stats)
