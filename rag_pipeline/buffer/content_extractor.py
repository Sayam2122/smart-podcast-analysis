# /podcast_rag_project/content_extractor.py

import logging
import ollama
from typing import List, Dict, Any, Optional
import json
import re

class ContentExtractor:
    """
    Extracts marketing-ready content from podcast data including quotes,
    insights, summaries, and social media assets.
    """
    
    def __init__(self, ollama_model: str = "mistral"):
        self.ollama_model = ollama_model
        self.llm_client = ollama.Client()
    
    def extract_key_quotes(self, segments: List[Dict[str, Any]], max_quotes: int = 10) -> List[Dict[str, Any]]:
        """
        Extract impactful quotes from podcast segments.
        
        Args:
            segments: List of enriched segment data
            max_quotes: Maximum number of quotes to extract
            
        Returns:
            List of quote dictionaries with metadata
        """
        logging.info(f"Extracting key quotes from {len(segments)} segments...")
        
        # Combine segments into larger text blocks for better context
        text_blocks = []
        for i in range(0, len(segments), 5):  # Group every 5 segments
            block_segments = segments[i:i+5]
            combined_text = " ".join([seg.get('text', '') for seg in block_segments])
            text_blocks.append({
                'text': combined_text,
                'start_time': block_segments[0].get('start_time', 0),
                'end_time': block_segments[-1].get('end_time', 0),
                'speaker': block_segments[0].get('speaker', 'Unknown'),
                'segments': block_segments
            })
        
        quotes = []
        for block in text_blocks:
            prompt = f"""
Analyze this podcast segment and extract 1-2 most impactful, quotable statements that would be valuable for social media or marketing.

Text: "{block['text']}"

For each quote, provide:
1. The exact quote (keep it concise, 10-50 words)
2. Why it's impactful (theme/insight)
3. Confidence score (1-10)

Format as JSON:
{{
  "quotes": [
    {{
      "text": "exact quote here",
      "impact_reason": "why this is quotable",
      "confidence": 8,
      "theme": "main theme/topic"
    }}
  ]
}}

Only return high-quality, standalone quotes that make sense without context.
"""
            
            try:
                response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
                result = response['response']
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    quote_data = json.loads(json_match.group())
                    for quote in quote_data.get('quotes', []):
                        if quote.get('confidence', 0) >= 6:  # Only high-confidence quotes
                            quote.update({
                                'start_time': block['start_time'],
                                'end_time': block['end_time'],
                                'speaker': block['speaker'],
                                'segment_ids': [seg.get('segment_id') for seg in block['segments']]
                            })
                            quotes.append(quote)
                            
            except Exception as e:
                logging.warning(f"Failed to extract quotes from block: {e}")
                continue
        
        # Sort by confidence and return top quotes
        quotes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return quotes[:max_quotes]
    
    def generate_social_assets(self, segments: List[Dict[str, Any]], episode_title: str = "") -> Dict[str, Any]:
        """
        Generate social media ready assets from podcast content.
        
        Args:
            segments: List of enriched segment data
            episode_title: Title of the episode
            
        Returns:
            Dictionary containing various social media assets
        """
        logging.info("Generating social media assets...")
        
        # Combine all text for analysis
        full_text = " ".join([seg.get('text', '') for seg in segments])
        
        prompt = f"""
Based on this podcast episode content, generate marketing-ready social media assets.

Episode Title: {episode_title}
Content: {full_text[:3000]}...

Generate the following assets in JSON format:

{{
  "summary": "2-3 sentence compelling summary",
  "taglines": ["3-5 catchy taglines for social media"],
  "headlines": ["3-5 attention-grabbing headlines"],
  "key_insights": ["3-5 main takeaways/insights"],
  "hashtags": ["relevant hashtags"],
  "qa_pairs": [
    {{"question": "What is...", "answer": "Concise answer..."}}
  ],
  "tweetable_quotes": ["Short quotes perfect for Twitter"],
  "linkedin_post": "Professional LinkedIn post text",
  "instagram_caption": "Engaging Instagram caption with hashtags"
}}

Make content engaging, professional, and shareable. Focus on value and insights.
"""
        
        try:
            response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
            result = response['response']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logging.warning("Could not parse social assets JSON")
                return {}
                
        except Exception as e:
            logging.error(f"Failed to generate social assets: {e}")
            return {}
    
    def extract_insights_and_themes(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract deep insights and themes from the podcast content.
        
        Args:
            segments: List of enriched segment data
            
        Returns:
            Dictionary containing insights and themes
        """
        logging.info("Extracting insights and themes...")
        
        # Group segments by speaker for speaker-specific insights
        speaker_segments = {}
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
        
        # Analyze overall themes
        full_text = " ".join([seg.get('text', '') for seg in segments])
        
        prompt = f"""
Analyze this podcast content and extract deep insights and themes.

Content: {full_text[:4000]}...

Provide analysis in JSON format:

{{
  "main_themes": ["primary themes discussed"],
  "key_concepts": ["important concepts/ideas"],
  "speaker_insights": {{
    "SPEAKER_00": "What this speaker focuses on and their perspective"
  }},
  "emotional_arc": "How emotions/tone evolve through the episode",
  "practical_takeaways": ["actionable insights listeners can apply"],
  "philosophical_points": ["deeper philosophical or conceptual insights"],
  "episode_significance": "Why this episode matters/what makes it valuable"
}}

Focus on substantial, meaningful insights rather than surface-level observations.
"""
        
        try:
            response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
            result = response['response']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group())
                
                # Add segment-level emotion analysis
                emotions = [seg.get('text_emotion', {}).get('emotion', 'neutral') for seg in segments]
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                insights['emotion_distribution'] = emotion_counts
                return insights
            else:
                logging.warning("Could not parse insights JSON")
                return {}
                
        except Exception as e:
            logging.error(f"Failed to extract insights: {e}")
            return {}
