# /podcast_rag_project/enhanced_query_engine.py

import ollama
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class EnhancedQueryEngine:
    """
    Enhanced query processing engine with advanced RAG capabilities, 
    conversation management, and content extraction features.
    """

    def __init__(self, vector_store, ollama_model: str, conversation_manager=None, content_extractor=None):
        """
        Initializes the Enhanced QueryEngine.

        Args:
            vector_store (VectorStore): An initialized VectorStore object for the selected episode.
            ollama_model (str): The name of the Ollama model to use for generation.
            conversation_manager: Manager for conversation history and context
            content_extractor: Extractor for marketing content and insights
        """
        self.vector_store = vector_store
        self.ollama_model = ollama_model
        self.llm_client = ollama.Client()
        self.conversation_manager = conversation_manager
        self.content_extractor = content_extractor
        
        # Enhanced configuration
        self.confidence_threshold = 0.7
        self.max_context_sources = 8
        self.response_cache = {}
        
        # Verify model availability at startup
        self._verify_model_availability()
        
    def _verify_model_availability(self):
        """
        Verify that the specified Ollama model is available and accessible.
        """
        try:
            # List available models
            models_response = self.llm_client.list()
            
            # Handle different response formats
            if hasattr(models_response, 'models'):
                # Response has 'models' attribute (list of model objects)
                available_model_names = [model.model for model in models_response.models]
            elif hasattr(models_response, '__iter__'):
                # Response is iterable (list of model objects)
                available_model_names = [model.model for model in models_response]
            else:
                # Response is a dict with 'models' key
                available_model_names = [model['model'] for model in models_response['models']]
            
            if self.ollama_model not in available_model_names:
                logging.warning(f"Model '{self.ollama_model}' not found. Available models: {available_model_names}")
                
                # Try to find a suitable fallback
                if available_model_names:
                    fallback_model = available_model_names[0]
                    logging.info(f"Using fallback model: {fallback_model}")
                    self.ollama_model = fallback_model
                else:
                    raise Exception("No Ollama models available")
            else:
                logging.info(f"Verified model '{self.ollama_model}' is available")
                
        except Exception as e:
            logging.error(f"Failed to verify Ollama model: {e}")
            logging.warning("Skipping model verification - will try to use specified model anyway")
            # Don't raise exception, just log and continue
            logging.info(f"Proceeding with model: {self.ollama_model}")
        
    def ask(self, question: str, user_id: str = "default") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhanced query processing with context awareness and grounded responses.
        
        Args:
            question (str): User's question
            user_id (str): User identifier for personalization
            
        Returns:
            Tuple of (answer, sources_used)
        """
        logging.info(f"Processing enhanced query: {question}")
        
        # Get conversation context
        conversation_context = ""
        user_interests = []
        if self.conversation_manager:
            conversation_context = self.conversation_manager.get_conversation_context()
            user_interests = self.conversation_manager.get_user_interests()
        
        # Enhanced semantic search with context awareness
        search_results = self._enhanced_search(question, conversation_context, user_interests)
        
        if not search_results:
            no_data_response = "I couldn't find relevant information about that topic in this episode. Could you try rephrasing your question or asking about a different aspect of the episode content?"
            
            if self.conversation_manager:
                self.conversation_manager.add_interaction(question, no_data_response, [])
            
            return no_data_response, []
        
        # Format context with enhanced structure
        context_str = self._format_enhanced_context(search_results)
        
        # Generate response with enhanced prompt, handle LLM/model errors gracefully
        try:
            answer = self._generate_enhanced_response(question, context_str, conversation_context, user_interests)
        except Exception as e:
            logging.error(f"LLM/model error: {e}")
            answer = (
                "⚠️ The language model could not process your request due to a system or memory error. "
                "Please try again later, restart Ollama, free up GPU memory, or use a smaller model."
            )

        # Add interaction to conversation manager
        if self.conversation_manager:
            query_metadata = {
                'user_interests': user_interests,
                'sources_count': len(search_results),
                'has_conversation_context': bool(conversation_context)
            }
            self.conversation_manager.add_interaction(question, answer, search_results, query_metadata)

        return answer, search_results
    
    def _enhanced_search(self, question: str, conversation_context: str, user_interests: List[str]) -> List[Dict[str, Any]]:
        """
        Enhanced search that considers conversation context and user interests.
        
        Args:
            question: User's current question
            conversation_context: Previous conversation context
            user_interests: User's identified interests
            
        Returns:
            List of relevant search results
        """
        # Primary search with original question
        primary_results = self.vector_store.search(question, k=self.max_context_sources)
        
        # If we have conversation context, do a contextual search
        contextual_results = []
        if conversation_context:
            # Extract key terms from conversation history
            context_query = f"{question} {' '.join(user_interests[:3])}"
            contextual_results = self.vector_store.search(context_query, k=4)
        
        # Combine and deduplicate results
        all_results = primary_results + contextual_results
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            result_id = tuple(result.get('source_segment_ids', []))
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by relevance (could implement more sophisticated scoring)
        return unique_results[:self.max_context_sources]
    
    def _format_enhanced_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Format search results into enhanced, structured context for the LLM.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant context found in the episode data."

        context_parts = ["=== PODCAST EPISODE CONTEXT ===\n"]
        
        # Group results by speaker and timestamp for better organization
        organized_results = self._organize_results_by_context(search_results)
        
        for i, (context_group, results) in enumerate(organized_results.items(), 1):
            context_parts.append(f"\n[CONTEXT SECTION {i}: {context_group}]")
            
            for result in results:
                timestamp = f"{result.get('start_time', 0):.1f}s - {result.get('end_time', 0):.1f}s"
                speaker = result.get('primary_speaker', result.get('speaker', 'Unknown'))
                emotion = result.get('dominant_emotion', result.get('text_emotion', {}).get('emotion', 'neutral'))
                
                context_parts.append(f"\n📍 TIME: {timestamp} | SPEAKER: {speaker} | TONE: {emotion}")
                context_parts.append(f"💬 CONTENT: \"{result.get('text', '')}\"")
                
                # Add chunk metadata if available
                if result.get('chunk_type'):
                    context_parts.append(f"📋 TYPE: {result.get('chunk_type', 'informational')}")
                
                # Add block summary if available
                if result.get('block_summary'):
                    context_parts.append(f"📝 SUMMARY: {result.get('block_summary', '')}")
                
                context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _organize_results_by_context(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize search results by contextual groupings for better structure.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Dictionary mapping context groups to results
        """
        organized = {}
        
        for result in results:
            # Group by speaker and time proximity
            speaker = result.get('primary_speaker', result.get('speaker', 'Unknown'))
            start_time = result.get('start_time', 0)
            
            # Create time-based groups (every 2 minutes)
            time_group = int(start_time // 120)  # 2-minute groups
            
            group_key = f"{speaker} - Section {time_group + 1}"
            
            if group_key not in organized:
                organized[group_key] = []
            
            organized[group_key].append(result)
        
        # Sort results within each group by timestamp
        for group in organized.values():
            group.sort(key=lambda x: x.get('start_time', 0))
        
        return organized
    
    def _generate_enhanced_response(self, question: str, context: str, conversation_history: str, user_interests: List[str]) -> str:
        """
        Generate enhanced response using improved prompt engineering.
        
        Args:
            question: User's question
            context: Formatted episode context
            conversation_history: Previous conversation context
            user_interests: User's interests for personalization
            
        Returns:
            Generated response
        """
        # Build personalization context
        personalization = ""
        if user_interests:
            personalization = f"\nUSER INTERESTS: {', '.join(user_interests[:5])}"
        
        # Enhanced prompt with strict grounding requirements
        prompt = f"""You are a professional podcast analyst with expertise in extracting precise insights from episode content. Your mission is to provide confident, data-backed responses based EXCLUSIVELY on the provided podcast context.

**CRITICAL INSTRUCTIONS:**
1. **ABSOLUTE GROUNDING**: Every statement must be directly supported by the provided episode context
2. **CONFIDENT TONE**: Use assertive, data-driven language (e.g., "The episode reveals...", "According to the discussion...", "The speaker emphasizes...")
3. **NO SPECULATION**: Never add information not present in the context
4. **PRECISE CITATIONS**: Reference specific speakers, timestamps, or content sections when possible
5. **SYNTHESIZE INSIGHTS**: Combine information from multiple context sections to provide comprehensive answers

**RESPONSE FRAMEWORK:**
- Lead with a confident statement answering the question
- Support with specific evidence from the episode
- Include relevant quotes or paraphrases when valuable
- End with additional insights or implications if supported by the data

{context}

{conversation_history}
{personalization}

**CURRENT QUESTION:** {question}

**CONFIDENT, DATA-GROUNDED RESPONSE:**"""

        try:
            response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
            answer = response['response'].strip()
            
            # Post-process response to ensure quality
            answer = self._post_process_response(answer)
            
            return answer
            
        except Exception as e:
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                logging.error(f"Model '{self.ollama_model}' not found: {e}")
                return f"❌ Model '{self.ollama_model}' is not available. Please install it with: ollama pull {self.ollama_model}"
            elif "connection" in error_msg or "refused" in error_msg:
                logging.error(f"Cannot connect to Ollama: {e}")
                return "❌ Cannot connect to Ollama. Please ensure Ollama is running (try: ollama serve)"
            else:
                logging.error(f"Failed to generate response: {e}")
                return "❌ I encountered an error while processing your question. Please try again or rephrase your query."
    
    def _post_process_response(self, response: str) -> str:
        """
        Post-process the response to ensure quality and consistency.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned and enhanced response
        """
        # Remove common hedging language that reduces confidence
        hedging_patterns = [
            "it seems like", "it appears that", "it might be", "possibly",
            "perhaps", "maybe", "it could be", "it's possible that"
        ]
        
        response_lower = response.lower()
        for pattern in hedging_patterns:
            if pattern in response_lower:
                # Replace with more confident alternatives
                response = response.replace(pattern, "")
                response = response.replace(pattern.title(), "")
        
        # Ensure response starts with a confident statement
        if response.startswith(("I think", "I believe", "In my opinion")):
            response = response.split(".", 1)[1].strip() if "." in response else response
        
        # Clean up extra whitespace and formatting
        response = " ".join(response.split())
        
        return response
    
    def extract_episode_content(self, episode_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive content assets from the episode.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Dictionary containing various content assets
        """
        if not self.content_extractor:
            return {"error": "Content extraction not enabled"}
        
        logging.info(f"Extracting content assets for episode: {episode_id}")
        
        # Get all segments from vector store
        all_segments = self.vector_store.metadata if hasattr(self.vector_store, 'metadata') else []
        
        if not all_segments:
            return {"error": "No episode data available for content extraction"}
        
        # Extract various types of content
        content_assets = {}
        
        try:
            # Extract key quotes
            content_assets['quotes'] = self.content_extractor.extract_key_quotes(all_segments)
            
            # Generate social media assets
            content_assets['social_assets'] = self.content_extractor.generate_social_assets(
                all_segments, f"Episode {episode_id}"
            )
            
            # Extract insights and themes
            content_assets['insights'] = self.content_extractor.extract_insights_and_themes(all_segments)
            
            # Save assets to file
            self._save_content_assets(episode_id, content_assets)
            
            logging.info(f"Successfully extracted content assets for {episode_id}")
            return content_assets
            
        except Exception as e:
            logging.error(f"Failed to extract content assets: {e}")
            return {"error": f"Content extraction failed: {str(e)}"}
    
    def _save_content_assets(self, episode_id: str, assets: Dict[str, Any]) -> None:
        """Save extracted content assets to file."""
        from config import CONTENT_ASSETS_DIR
        
        os.makedirs(CONTENT_ASSETS_DIR, exist_ok=True)
        
        assets_file = os.path.join(CONTENT_ASSETS_DIR, f"{episode_id}_content_assets.json")
        
        # Add metadata
        assets['metadata'] = {
            'episode_id': episode_id,
            'extraction_timestamp': datetime.now().isoformat(),
            'extractor_version': '2.0'
        }
        
        with open(assets_file, 'w', encoding='utf-8') as f:
            json.dump(assets, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved content assets to {assets_file}")
    
    def generate_followup_suggestions(self, current_response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """
        Generate intelligent follow-up question suggestions.
        
        Args:
            current_response: The current AI response
            sources: Sources used in the response
            
        Returns:
            List of suggested follow-up questions
        """
        if self.conversation_manager:
            return self.conversation_manager.suggest_followup_questions(current_response, sources)
        
        # Fallback suggestions based on sources
        suggestions = []
        if sources:
            for source in sources[:2]:
                speaker = source.get('primary_speaker', source.get('speaker', 'the speaker'))
                suggestions.append(f"What else does {speaker} discuss in this episode?")
                
                if source.get('block_summary'):
                    suggestions.append("Can you elaborate on this topic further?")
        
        return suggestions[:3]
    
    def get_conversation_analysis(self) -> Dict[str, Any]:
        """Get analysis of current conversation session."""
        if self.conversation_manager:
            return self.conversation_manager.analyze_conversation_patterns()
        return {}
    
    def save_session(self) -> None:
        """Save current conversation session."""
        if self.conversation_manager:
            self.conversation_manager.save_session()
