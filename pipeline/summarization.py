"""
Summarization module for the podcast analysis pipeline.
Uses local Ollama LLM to generate summaries for semantic blocks.
"""

import time
import json
import requests
from typing import List, Dict, Optional, Tuple
import re

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class PodcastSummarizer:
    """
    Local LLM-based summarization using Ollama
    Generates summaries, key points, and insights for semantic blocks
    """
    
    def __init__(self,
                 model_name: str = "mistral:7b",
                 ollama_url: str = "http://localhost:11434",
                 max_tokens: int = 300,
                 temperature: float = 0.3):
        """
        Initialize podcast summarizer
        
        Args:
            model_name: Ollama model to use (mistral:7b, llama3.2:3b, etc.)
            ollama_url: Ollama server URL
            max_tokens: Maximum tokens for summaries
            temperature: Temperature for generation (0.0-1.0)
        """
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.file_utils = get_file_utils()
        
        # Check Ollama availability
        self.ollama_available = self._check_ollama_connection()
        
        logger.info(f"Initializing summarizer | "
                   f"Model: {model_name} | "
                   f"Ollama: {'Available' if self.ollama_available else 'Not Available'}")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model.get('name', '') for model in models]
                
                if self.model_name in available_models:
                    logger.info(f"Ollama server connected | Model {self.model_name} available")
                    return True
                else:
                    logger.warning(f"Model {self.model_name} not found | Available: {available_models}")
                    # Try to pull the model
                    return self._pull_model()
            else:
                logger.warning(f"Ollama server not responding properly: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.warning(f"Cannot connect to Ollama server: {e}")
            return False
    
    def _pull_model(self) -> bool:
        """Attempt to pull the model if not available"""
        try:
            logger.info(f"Attempting to pull model: {self.model_name}")
            
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes for model download
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {self.model_name}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def _generate_with_ollama(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Generate text using Ollama API"""
        if not self.ollama_available:
            return None
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return None
    
    def summarize_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Summarize all semantic blocks with emotion analysis
        
        Args:
            blocks: List of semantic blocks from segmentation
            
        Returns:
            List of blocks with added summary information
        """
        logger.info(f"Starting summarization for {len(blocks)} blocks")
        
        if not self.ollama_available:
            logger.warning("Ollama not available, using fallback summarization")
            return self._fallback_summarization(blocks)
        
        start_time = time.time()
        summarized_blocks = []
        
        for i, block in enumerate(blocks):
            logger.info(f"Summarizing block {i+1}/{len(blocks)}")
            
            # Analyze emotions in the block segments
            emotion_analysis = self._analyze_block_emotions(block)
            
            # Generate summary
            summary = self._summarize_block(block)
            
            # Generate key points
            key_points = self._extract_key_points(block)
            
            # Generate insights with emotion context
            insights = self._extract_insights(block, emotion_analysis)
            
            # Enhance block with all analysis
            enhanced_block = block.copy()
            enhanced_block.update({
                'summary': summary,
                'key_points': key_points,
                'insights': insights,
                'emotion_analysis': emotion_analysis,
                'summary_stats': {
                    'original_length': len(block['text']),
                    'summary_length': len(summary) if summary else 0,
                    'compression_ratio': len(summary) / len(block['text']) if block['text'] and summary else 0,
                    'segments_analyzed': len(block.get('segments', [])),
                    'dominant_emotion': emotion_analysis.get('dominant_emotion', 'neutral'),
                    'emotion_confidence': emotion_analysis.get('confidence', 0.0)
                }
            })
            
            summarized_blocks.append(enhanced_block)
        
        processing_time = time.time() - start_time
        logger.info(f"Summarization completed in {processing_time:.2f}s")
        
        return summarized_blocks
    
    def _analyze_block_emotions(self, block: Dict) -> Dict:
        """Analyze emotions across all segments in a block"""
        segments = block.get('segments', [])
        if not segments:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'emotion_distribution': {},
                'segment_count': 0
            }
        
        # Collect all emotions from segments
        all_emotions = []
        emotion_scores = {}
        valid_segments = 0
        
        for segment in segments:
            # Try different emotion data locations
            emotion_data = None
            
            # Check for combined emotion first
            if 'emotions' in segment and isinstance(segment['emotions'], dict):
                combined = segment['emotions'].get('combined_emotion', {})
                if isinstance(combined, dict) and 'emotion' in combined:
                    emotion_data = combined
            
            # Fall back to text emotion
            if not emotion_data and 'text_emotion' in segment:
                text_emotion = segment['text_emotion']
                if isinstance(text_emotion, dict) and 'emotion' in text_emotion:
                    emotion_data = text_emotion
            
            # Fall back to audio emotion
            if not emotion_data and 'audio_emotion' in segment:
                audio_emotion = segment['audio_emotion']
                if isinstance(audio_emotion, dict) and 'emotion' in audio_emotion:
                    emotion_data = audio_emotion
            
            if emotion_data:
                emotion = emotion_data.get('emotion')
                if emotion:
                    all_emotions.append(emotion)
                    valid_segments += 1
                    
                    # Aggregate emotion scores
                    scores = emotion_data.get('all_scores', {})
                    for emotion_type, score in scores.items():
                        if emotion_type not in emotion_scores:
                            emotion_scores[emotion_type] = []
                        emotion_scores[emotion_type].append(float(score))
        
        if not all_emotions:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'emotion_distribution': {},
                'segment_count': len(segments),
                'analyzed_segments': 0
            }
        
        # Calculate dominant emotion
        from collections import Counter
        emotion_counts = Counter(all_emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Calculate average scores for each emotion
        avg_emotion_scores = {}
        for emotion_type, scores in emotion_scores.items():
            avg_emotion_scores[emotion_type] = sum(scores) / len(scores)
        
        # Calculate confidence as the average score of the dominant emotion
        confidence = avg_emotion_scores.get(dominant_emotion, 0.5)
        
        # Create distribution
        total_emotions = len(all_emotions)
        emotion_distribution = {
            emotion: count / total_emotions 
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'emotion_distribution': emotion_distribution,
            'average_scores': avg_emotion_scores,
            'segment_count': len(segments),
            'analyzed_segments': valid_segments,
            'emotion_variety': len(emotion_counts)
        }
    
    def _summarize_block(self, block: Dict) -> str:
        """Generate summary for a single block"""
        text = block['text']
        duration = block.get('duration', 0)
        
        # Create context-aware prompt
        system_prompt = """You are an expert at summarizing podcast segments. 
Create concise, informative summaries that capture the main ideas and key information.
Focus on the core message and important details."""
        
        prompt = f"""Please summarize this podcast segment in 2-3 sentences:

Duration: {duration:.1f} seconds
Segment: {len(block.get('segments', []))} parts

Text:
{text}

Summary:"""
        
        summary = self._generate_with_ollama(prompt, system_prompt)
        
        if not summary:
            # Fallback to extractive summary
            return self._extractive_summary(text)
        
        return summary
    
    def _extract_key_points(self, block: Dict) -> List[str]:
        """Extract key points from a block"""
        text = block['text']
        
        system_prompt = """You are an expert at identifying key points in conversations.
Extract the most important points as a bullet list.
Focus on actionable insights, important facts, and main arguments."""
        
        prompt = f"""Extract 3-5 key points from this podcast segment:

Text:
{text}

Key Points (return as numbered list):"""
        
        response = self._generate_with_ollama(prompt, system_prompt)
        
        if response:
            # Parse numbered list
            points = []
            for line in response.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    points.append(re.sub(r'^\d+\.\s*', '', line))
                elif line.startswith('- '):
                    points.append(line[2:])
                elif line.startswith('• '):
                    points.append(line[2:])
            return points[:5]  # Limit to 5 points
        
        # Fallback key point extraction
        return self._fallback_key_points(text)
    
    def _extract_insights(self, block: Dict, emotion_analysis: Dict = None) -> Dict:
        """Extract insights from a block with emotion context"""
        if not self.ollama_available:
            return self._fallback_insights(block)
        
        # Include emotion context in the analysis prompt
        emotion_context = ""
        if emotion_analysis:
            dominant_emotion = emotion_analysis.get('dominant_emotion', 'neutral')
            confidence = emotion_analysis.get('confidence', 0.0)
            emotion_variety = emotion_analysis.get('emotion_variety', 1)
            
            emotion_context = f"""
Emotional Context:
- Dominant emotion: {dominant_emotion} (confidence: {confidence:.2f})
- Emotional variety: {emotion_variety} different emotions detected
- Emotion distribution: {emotion_analysis.get('emotion_distribution', {})}
"""
        
        prompt = f"""Analyze this podcast segment and provide insights:

Text: {block['text'][:2000]}...
Duration: {block.get('duration', 0):.1f} seconds
Segments: {len(block.get('segments', []))} parts
{emotion_context}

Provide insights in this format:
Theme: [main theme]
Significance: [why this is important]
Context: [relevant context or background]
Emotional tone: [how emotion affects the content]
"""
        
        response = self._generate_with_ollama(prompt)
        
        insights = {
            'theme': 'General discussion',
            'significance': 'Part of broader conversation',
            'context': 'Conversational segment',
            'emotional_tone': emotion_analysis.get('dominant_emotion', 'neutral') if emotion_analysis else 'neutral'
        }
        
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('theme:'):
                    insights['theme'] = re.sub(r'^.*?:', '', line).strip()
                elif line.lower().startswith('significance:'):
                    insights['significance'] = re.sub(r'^.*?:', '', line).strip()
                elif line.lower().startswith('context:'):
                    insights['context'] = re.sub(r'^.*?:', '', line).strip()
                elif line.lower().startswith('emotional tone:'):
                    insights['emotional_tone'] = re.sub(r'^.*?:', '', line).strip()
        
        return insights
    
    def _parse_insights(self, response: str) -> Dict:
        """Parse insights from LLM response"""
        insights = {
            'theme': 'General discussion',
            'sentiment': 'neutral',
            'significance': 'Part of ongoing conversation'
        }
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            line_lower = line.lower()
            if 'theme' in line_lower or 'topic' in line_lower:
                insights['theme'] = re.sub(r'^.*?:', '', line).strip()
            elif 'sentiment' in line_lower:
                sentiment_text = re.sub(r'^.*?:', '', line).strip().lower()
                if 'positive' in sentiment_text:
                    insights['sentiment'] = 'positive'
                elif 'negative' in sentiment_text:
                    insights['sentiment'] = 'negative'
                else:
                    insights['sentiment'] = 'neutral'
            elif 'significance' in line_lower or 'takeaway' in line_lower:
                insights['significance'] = re.sub(r'^.*?:', '', line).strip()
        
        return insights
    
    def _fallback_summarization(self, blocks: List[Dict]) -> List[Dict]:
        """Fallback summarization when Ollama is not available, with emotion analysis"""
        logger.info("Using fallback summarization methods")
        
        summarized_blocks = []
        
        for block in blocks:
            # Analyze emotions even in fallback mode
            emotion_analysis = self._analyze_block_emotions(block)
            
            # Extractive summary
            summary = self._extractive_summary(block['text'])
            
            # Simple key points
            key_points = self._fallback_key_points(block['text'])
            
            # Basic insights with emotion
            insights = self._fallback_insights(block)
            insights['emotional_tone'] = emotion_analysis.get('dominant_emotion', 'neutral')
            
            enhanced_block = block.copy()
            enhanced_block.update({
                'summary': summary,
                'key_points': key_points,
                'insights': insights,
                'emotion_analysis': emotion_analysis,
                'summary_stats': {
                    'original_length': len(block['text']),
                    'summary_length': len(summary),
                    'compression_ratio': len(summary) / len(block['text']) if block['text'] else 0,
                    'segments_analyzed': len(block.get('segments', [])),
                    'dominant_emotion': emotion_analysis.get('dominant_emotion', 'neutral'),
                    'emotion_confidence': emotion_analysis.get('confidence', 0.0)
                }
            })
            
            summarized_blocks.append(enhanced_block)
        
        return summarized_blocks
    
    def _extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """Create extractive summary by selecting important sentences"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Favor longer sentences and those at beginning/end
            position_score = 1.0 if i < 2 or i >= len(sentences) - 2 else 0.5
            length_score = min(len(sentence.split()) / 20, 1.0)  # Normalize by 20 words
            total_score = position_score * 0.6 + length_score * 0.4
            
            scored_sentences.append((sentence, total_score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scored_sentences[:max_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in selected:
                summary_sentences.append(sentence)
        
        return '. '.join(summary_sentences) + '.'
    
    def _fallback_key_points(self, text: str) -> List[str]:
        """Extract key points using simple methods"""
        # Look for sentences with question words or emphatic phrases
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Look for important indicators
            if any(phrase in sentence.lower() for phrase in [
                'important', 'key', 'main', 'critical', 'essential',
                'remember', 'note that', 'keep in mind', 'the point is',
                'what matters', 'significant', 'crucial'
            ]):
                key_phrases.append(sentence)
            
            # Look for questions
            elif sentence.strip().endswith('?'):
                key_phrases.append(sentence)
        
        # If no special phrases found, use longest sentences
        if not key_phrases:
            long_sentences = sorted(sentences, key=len, reverse=True)
            key_phrases = [s.strip() for s in long_sentences[:3] if len(s.strip()) > 20]
        
        return key_phrases[:5]
    
    def _fallback_insights(self, block: Dict) -> Dict:
        """Generate basic insights without LLM"""
        text = block['text'].lower()
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'problem', 'issue', 'wrong']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Extract topic from key topics
        theme = ', '.join(block.get('key_topics', ['General discussion'])[:3])
        
        return {
            'theme': theme,
            'sentiment': sentiment,
            'significance': f"Discussion segment covering {theme}"
        }
    
    def generate_overall_summary(self, blocks: List[Dict]) -> Dict:
        """
        Generate overall summary of the entire podcast
        
        Args:
            blocks: List of summarized blocks
            
        Returns:
            Overall summary with key insights
        """
        logger.info("Generating overall podcast summary")
        
        # Collect all summaries
        block_summaries = [block.get('summary', '') for block in blocks if block.get('summary')]
        combined_summaries = ' '.join(block_summaries)
        
        # Collect all key points
        all_key_points = []
        for block in blocks:
            all_key_points.extend(block.get('key_points', []))
        
        # Generate overall summary
        if self.ollama_available and combined_summaries:
            overall_summary = self._generate_overall_with_llm(combined_summaries, all_key_points)
        else:
            overall_summary = self._generate_overall_fallback(blocks)
        
        return overall_summary
    
    def _generate_overall_with_llm(self, summaries: str, key_points: List[str]) -> Dict:
        """Generate overall summary using LLM"""
        system_prompt = """You are an expert at creating comprehensive podcast summaries.
Focus on the actual content, themes, and key insights discussed in the podcast.
Do not mention technical details like duration, number of speakers, or file information."""
        
        prompt = f"""Create an overall summary for this podcast based on the segment summaries:

Segment Summaries:
{summaries}

Key Points:
{chr(10).join(f"- {point}" for point in key_points[:10])}

Please provide:
1. A 2-3 sentence overall summary that captures what the podcast is actually about
2. Top 3 main themes or topics discussed
3. Key takeaways or insights shared

Focus on the content and meaning, not technical aspects.

Overall Summary:"""
        
        response = self._generate_with_ollama(prompt, system_prompt)
        
        if response:
            return self._parse_overall_summary(response)
        else:
            return self._generate_overall_fallback([])  # Empty list as fallback
    
    def _parse_overall_summary(self, response: str) -> Dict:
        """Parse overall summary response"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        overall = {
            'summary': '',
            'main_themes': [],
            'key_takeaways': []
        }
        
        current_section = 'summary'
        
        for line in lines:
            line_lower = line.lower()
            if 'theme' in line_lower:
                current_section = 'themes'
                continue
            elif 'takeaway' in line_lower:
                current_section = 'takeaways'
                continue
            
            if current_section == 'summary' and not overall['summary']:
                overall['summary'] = line
            elif current_section == 'themes' and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                theme = re.sub(r'^[-•\d\.\s]+', '', line)
                overall['main_themes'].append(theme)
            elif current_section == 'takeaways' and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                takeaway = re.sub(r'^[-•\d\.\s]+', '', line)
                overall['key_takeaways'].append(takeaway)
        
        return overall
    
    def _generate_overall_fallback(self, blocks: List[Dict]) -> Dict:
        """Generate overall summary without LLM"""
        # Collect themes
        all_themes = []
        for block in blocks:
            all_themes.extend(block.get('key_topics', []))
        
        # Count theme frequency
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Get top themes
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        main_themes = [theme for theme, count in top_themes]
        
        # Generate basic summary
        total_duration = sum(block.get('duration', 0) for block in blocks)
        
        return {
            'summary': f"Podcast discussion covering {len(blocks)} main segments over {total_duration/60:.1f} minutes, focusing on {', '.join(main_themes[:2])}.",
            'main_themes': main_themes,
            'key_takeaways': [
                f"Discussion included {len(blocks)} distinct segments",
                f"Main topics: {', '.join(main_themes[:3])}",
                f"Total duration: {total_duration/60:.1f} minutes"
            ]
        }
    
    def save_summaries(self, summarized_blocks: List[Dict], output_path: str) -> None:
        """
        Save summarized blocks to JSON file
        
        Args:
            summarized_blocks: List of blocks with summaries
            output_path: Path to save the results
        """
        # Generate overall summary
        overall_summary = self.generate_overall_summary(summarized_blocks)
        
        summary_data = {
            'metadata': {
                'model_used': self.model_name,
                'total_blocks': len(summarized_blocks),
                'total_duration': sum(block.get('duration', 0) for block in summarized_blocks),
                'ollama_available': self.ollama_available
            },
            'overall_summary': overall_summary,
            'blocks': summarized_blocks
        }
        
        self.file_utils.save_json(summary_data, output_path)
        logger.info(f"Saved summaries to: {output_path}")


def summarize_semantic_blocks(blocks: List[Dict],
                            model_name: str = "mistral:7b",
                            ollama_url: str = "http://localhost:11434") -> List[Dict]:
    """
    Convenience function to summarize semantic blocks
    
    Args:
        blocks: List of semantic blocks
        model_name: Ollama model to use
        ollama_url: Ollama server URL
        
    Returns:
        List of summarized blocks
    """
    summarizer = PodcastSummarizer(
        model_name=model_name,
        ollama_url=ollama_url
    )
    return summarizer.summarize_blocks(blocks)
