import time
from typing import List, Dict, Any, Optional
import ollama
import re
from collections import Counter

class RelationSynthesis:
    """
    Module for finding relationships, themes, and cross-block/segment patterns using LLM and analytics.
    """
    def __init__(self, llm_client, analytics=None):
        self.llm_client = llm_client
        self.analytics = analytics

    def analyze(self, segments: List[Dict[str, Any]], question: str) -> str:
        # Use analytics for quick theme/relationship extraction
        themes = self._extract_themes(segments)
        speakers = self._extract_speakers(segments)
        emotion_trends = self._extract_emotion_trends(segments)
        # LLM prompt for deeper relationship analysis
        rel_context = []
        for seg in segments:
            rel_context.append(f"Block {seg.get('block_id')}, Segment {seg.get('segment_id')}: {seg.get('text', '')}")
        rel_prompt = f"""
You are an expert podcast analyst. Given the following podcast segments, analyze and describe:
- Any relationships, recurring themes, or patterns across blocks/segments
- How the segments relate to the user's question: "{question}"
- Any cross-segment insights, contradictions, or developments
- Thematic trends: {themes}
- Speaker patterns: {speakers}
- Emotional trends: {emotion_trends}

Segments:
{chr(10).join(rel_context)}

---
Provide a structured analysis of relationships, themes, and insights (not just a list of snippets):
"""
        try:
            response = self.llm_client.generate(model="mistral:latest", prompt=rel_prompt)
            return "\n=== RELATIONSHIPS & THEMES ===\n" + response['response'].strip()
        except Exception as e:
            return f"[RelationSynthesis] Relationship analysis error: {e}"

    def _extract_themes(self, segments):
        all_themes = []
        for seg in segments:
            if seg.get('block_key_points'):
                all_themes.extend(seg['block_key_points'])
            if seg.get('block_insights') and 'theme' in seg['block_insights']:
                all_themes.append(seg['block_insights']['theme'])
        theme_counts = Counter(all_themes)
        return theme_counts.most_common(5)

    def _extract_speakers(self, segments):
        speakers = [seg.get('speaker', 'Unknown') for seg in segments]
        speaker_counts = Counter(speakers)
        return speaker_counts.most_common(3)

    def _extract_emotion_trends(self, segments):
        emotions = [seg.get('text_emotion', {}).get('emotion', 'neutral') for seg in segments]
        emotion_counts = Counter(emotions)
        return emotion_counts.most_common(3)

class QueryEngine:
    """
    Agentic, expert-level query engine for Podcast RAG System v2.0.
    Multi-stage: retrieve, relate, synthesize, summarize, suggest follow-ups.
    Synthesizes answers with literal and emotional understanding using the episode model.
    Handles intent detection for quotes, assets, audio, and standard queries.
    """
    def __init__(self, vector_store, ollama_model: str, episode_model: Optional[Dict[str, Any]] = None, all_segments: Optional[List[Dict[str, Any]]] = None):
        self.vector_store = vector_store
        self.ollama_model = ollama_model
        self.llm_client = ollama.Client()
        self.conversation_history = []
        self.episode_model = episode_model or {}
        self.all_segments = all_segments or []
        self.token_limit = 3500  # For Mistral 7B, adjust as needed
        self.neighbor_window = 1  # Number of segments before/after to include
        self.relation_synth = RelationSynthesis(self.llm_client)
        self.domain_guidance = ''  # Set externally by CLI if available

    def ask(self, question: str, k: int = 8, intent: str = 'standard') -> Dict[str, Any]:
        start_time = time.time()
        # Intent handling (quotes/assets/audio handled as before)
        if intent == 'quotes':
            quotes = self.extract_quotes()
            elapsed = time.time() - start_time
            return {
                'answer': self._format_quotes_answer(quotes),
                'sources': quotes,
                'context': '',
                'suggestions': [],
                'processing_time': elapsed
            }
        if intent == 'assets':
            assets = self.extract_assets()
            elapsed = time.time() - start_time
            return {
                'answer': self._format_assets_answer(assets),
                'sources': assets,
                'context': '',
                'suggestions': [],
                'processing_time': elapsed
            }
        if intent == 'audio':
            audio_segments = self.extract_audio_segments()
            elapsed = time.time() - start_time
            return {
                'answer': self._format_audio_answer(audio_segments),
                'sources': audio_segments,
                'context': '',
                'suggestions': [],
                'processing_time': elapsed
            }
        # Multi-stage: retrieve, relate, synthesize, summarize, suggest
        results = self.vector_store.search(question, k)
        key_terms = self._extract_key_terms(question, results)
        expanded_segments = self._expand_context(results, key_terms)
        context_chunks = self._manage_token_budget(expanded_segments)
        # Confident presence/absence logic
        if not self._query_terms_present(question, context_chunks):
            closest = self._find_closest_content(question)
            elapsed = time.time() - start_time
            return {
                'answer': f"The term(s) '{question}' were not found in the selected episode(s). Closest related content: {closest}",
                'sources': [],
                'context': '',
                'suggestions': [],
                'processing_time': elapsed
            }
        # Stage 1: Context formatting (with relationships/themes)
        context_str = self._format_context(context_chunks)
        relation_str = self.relation_synth.analyze(context_chunks, question)
        episode_model_str = self._format_episode_model()
        history_str = self._format_history()
        # Stage 2: Synthesis prompt
        prompt = self._build_synthesis_prompt(question, context_str, relation_str, episode_model_str, history_str, self.domain_guidance)
        try:
            response = self.llm_client.generate(model=self.ollama_model, prompt=prompt)
            answer = response['response'].strip()
            # Stage 3: Summarize (if context is large)
            if len(context_chunks) > 10:
                summary_prompt = self._build_summary_prompt(answer, context_str, relation_str, episode_model_str, history_str, self.domain_guidance)
                summary_response = self.llm_client.generate(model=self.ollama_model, prompt=summary_prompt)
                answer = summary_response['response'].strip()
            # Stage 4: Meta-analysis
            meta_prompt = self._build_meta_prompt(question, answer, context_str, relation_str, episode_model_str, history_str, self.domain_guidance)
            meta_response = self.llm_client.generate(model=self.ollama_model, prompt=meta_prompt)
            final_answer = meta_response['response'].strip()
        except Exception as e:
            final_answer = f"[QueryEngine] Error: {e}"
        self.conversation_history.append({
            'user': question,
            'ai': final_answer,
            'sources': context_chunks,
            'timestamp': time.time()
        })
        suggestions = self._suggest_followups(question, final_answer, context_chunks)
        elapsed = time.time() - start_time
        return {
            'answer': final_answer,
            'sources': context_chunks,
            'context': context_str + '\n' + relation_str + '\n' + episode_model_str,
            'suggestions': suggestions,
            'processing_time': elapsed
        }

    def _extract_key_terms(self, question: str, results: List[Dict[str, Any]]) -> List[str]:
        # Extract keywords from question and top segments
        words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        for seg in results:
            for kp in seg.get('block_key_points', []):
                words.add(kp.lower())
            if seg.get('block_insights') and 'theme' in seg['block_insights']:
                words.add(seg['block_insights']['theme'].lower())
        return list(words)

    def _expand_context(self, results: List[Dict[str, Any]], key_terms: List[str]) -> List[Dict[str, Any]]:
        # Start with initial results
        expanded = { (seg.get('block_id'), seg.get('segment_id')): seg for seg in results }
        # Add segments containing key terms
        for seg in self.all_segments:
            text = seg.get('text', '').lower()
            if any(term in text for term in key_terms):
                expanded[(seg.get('block_id'), seg.get('segment_id'))] = seg
        # Add neighbors for each top segment
        seg_list = self.all_segments
        seg_idx_map = { (seg.get('block_id'), seg.get('segment_id')): i for i, seg in enumerate(seg_list) }
        for seg in results:
            idx = seg_idx_map.get((seg.get('block_id'), seg.get('segment_id')))
            if idx is not None:
                for offset in range(-self.neighbor_window, self.neighbor_window+1):
                    nidx = idx + offset
                    if 0 <= nidx < len(seg_list):
                        nseg = seg_list[nidx]
                        expanded[(nseg.get('block_id'), nseg.get('segment_id'))] = nseg
        # Return as a list, sorted by start_time
        return sorted(expanded.values(), key=lambda s: s.get('start_time', 0))

    def _manage_token_budget(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Estimate tokens (1 token ~ 4 chars for English)
        total_tokens = sum(len(seg.get('text', '')) // 4 for seg in segments)
        if total_tokens <= self.token_limit:
            return segments
        # If too large, cluster or summarize
        # For now, just take the most relevant/emotional segments
        segments = sorted(segments, key=lambda s: max(s.get('text_emotion', {}).get('all_scores', {}).values() or [0]), reverse=True)
        return segments[:max(8, self.token_limit // 200)]

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant context found."
        lines = ["\n=== CONTEXT SOURCES (with Relationships & Themes) ==="]
        for i, seg in enumerate(results, 1):
            lines.append(f"\n[Source {i}]")
            lines.append(f"Block ID: {seg.get('block_id')} | Segment ID: {seg.get('segment_id')}")
            lines.append(f"Time: {seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s | Speaker: {seg.get('speaker', 'Unknown')}")
            lines.append(f"Relevance Score: {seg.get('relevance_score', 0):.4f}")
            lines.append(f"Word Count: {len(seg.get('text', '').split())} | Content Density: {seg.get('block_stats', {}).get('compression_ratio', 'N/A')}")
            text_em = seg.get('text_emotion', {})
            audio_em = seg.get('audio_emotion', {})
            lines.append(f"Text Emotion: {text_em.get('emotion', 'N/A')} (scores: {text_em.get('all_scores', {})})")
            lines.append(f"Audio Emotion: {audio_em.get('emotion', 'N/A')} (scores: {audio_em.get('all_scores', {})})")
            if seg.get('block_summary'):
                lines.append(f"Block Summary: {seg.get('block_summary')}")
            if seg.get('block_key_points'):
                lines.append(f"Key Points: {seg.get('block_key_points')}")
            if seg.get('block_insights'):
                lines.append(f"Insights: {seg.get('block_insights')}")
            # Highlight relationships/themes if present
            if 'theme' in seg.get('block_insights', {}):
                lines.append(f"Theme: {seg['block_insights']['theme']}")
            if 'significance' in seg.get('block_insights', {}):
                lines.append(f"Significance: {seg['block_insights']['significance']}")
            lines.append(f"Text: \"{seg.get('text', '')}\"")
            lines.append("---")
        return "\n".join(lines)

    def _format_episode_model(self) -> str:
        if not self.episode_model:
            return ""
        lines = ["\n=== EPISODE MODEL (Holistic Analysis) ==="]
        if self.episode_model.get('main_themes'):
            lines.append(f"Main Themes: {self.episode_model['main_themes']}")
        if self.episode_model.get('emotion_sequence'):
            lines.append(f"Emotional Arc: {self.episode_model['emotion_sequence'][:10]} ...")
        if self.episode_model.get('emotion_peaks'):
            lines.append(f"Emotional Peaks: {[m['segment_id'] for m in self.episode_model['emotion_peaks']]}")
        if self.episode_model.get('speaker_counts'):
            lines.append(f"Speaker Counts: {self.episode_model['speaker_counts']}")
        if self.episode_model.get('key_moments'):
            lines.append(f"Key Moments: {[m['segment_id'] for m in self.episode_model['key_moments']]}")
        return "\n".join(lines)

    def _format_history(self) -> str:
        if not self.conversation_history:
            return ""
        lines = ["--- Conversation History ---"]
        for h in self.conversation_history[-5:]:
            lines.append(f"User: {h['user']}\nAI: {h['ai']}")
        return "\n".join(lines)

    def _build_synthesis_prompt(self, question: str, context: str, relation_str: str, episode_model_str: str, history: str, domain_guidance: str = '') -> str:
        return f"""
You are an expert podcast analyst. Synthesize a robust, high-level answer to the user's question using ONLY the provided context, relationship analysis, and episode model. Your answer should:
- Demonstrate both literal and emotional understanding of the podcast
- Relate and synthesize information across blocks/segments
- Highlight recurring themes, emotional arcs, key moments, and developments
- Reference specific sources (block/segment/timestamp) where relevant
- Provide a comprehensive, insightful response
{f'\n[Domain Guidance]: {domain_guidance}' if domain_guidance else ''}

{history}
{context}
{relation_str}
{episode_model_str}

--- Current Question ---
User: {question}

--- Your Answer (Grounded in the context, relationships, and episode model above) ---
AI:"""

    def _build_summary_prompt(self, answer, context, relation_str, episode_model_str, history, domain_guidance: str = '') -> str:
        return f"""
You are an expert podcast analyst. Given the previous answer and all the context, summarize the key findings, relationships, and emotional/thematic developments in a concise, insightful way for a busy user.
{f'\n[Domain Guidance]: {domain_guidance}' if domain_guidance else ''}

Previous Answer:
{answer}

{history}
{context}
{relation_str}
{episode_model_str}

--- SUMMARY ---
AI:"""

    def _build_meta_prompt(self, question: str, prev_answer: str, context: str, relation_str: str, episode_model_str: str, history: str, domain_guidance: str = '') -> str:
        return f"""
You are an expert podcast analyst. Given your previous answer and all the provided context, perform a deeper meta-analysis:
- What additional, more nuanced, or hidden insights can you provide?
- Are there any subtle emotional or thematic developments not previously mentioned?
- Synthesize a final, expert-level answer that would help a listener, creator, or researcher fully understand the podcast's meaning and emotional arc.
{f'\n[Domain Guidance]: {domain_guidance}' if domain_guidance else ''}

Previous Answer:
{prev_answer}

{history}
{context}
{relation_str}
{episode_model_str}

--- Current Question ---
User: {question}

--- FINAL, DEEPER ANSWER ---
AI:"""

    def _suggest_followups(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> List[str]:
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

    def extract_quotes(self) -> List[Dict[str, Any]]:
        quotes = []
        for seg in self.all_segments:
            text = seg.get('text', '')
            if re.search(r'[\u0900-\u097F]', text) or 'quote' in text.lower() or 'shloka' in text.lower():
                quotes.append({
                    'text': text,
                    'timestamp': f"{seg.get('start_time', 0):.1f}s",
                    'speaker': seg.get('speaker', 'Unknown'),
                    'block_id': seg.get('block_id'),
                    'segment_id': seg.get('segment_id')
                })
        return quotes

    def extract_assets(self) -> List[Dict[str, Any]]:
        # Use content generator for social media assets
        from content_generator import ContentGenerator
        return ContentGenerator.social_media_posts(self.all_segments)

    def extract_audio_segments(self) -> List[Dict[str, Any]]:
        # For demo: return the most relevant/emotional segments
        segments = sorted(self.all_segments, key=lambda s: max(s.get('text_emotion', {}).get('all_scores', {}).values() or [0]), reverse=True)
        return segments[:10]

    def _format_quotes_answer(self, quotes: List[Dict[str, Any]]) -> str:
        if not quotes:
            return "No quotes or shlokas found in the selected episode(s)."
        lines = ["Quotes/Shlokas found:"]
        for q in quotes:
            lines.append(f"[{q['timestamp']}] {q['speaker']}: {q['text']}")
        return '\n'.join(lines)

    def _format_assets_answer(self, assets: List[Dict[str, Any]]) -> str:
        if not assets:
            return "No social media assets found in the selected episode(s)."
        lines = ["Social Media Posts:"]
        for a in assets:
            lines.append(str(a))
        return '\n'.join(lines)

    def _format_audio_answer(self, audio_segments: List[Dict[str, Any]]) -> str:
        if not audio_segments:
            return "No relevant segments found for audio generation."
        lines = ["Segments for audio generation:"]
        for seg in audio_segments:
            lines.append(f"[{seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s] {seg.get('speaker', 'Unknown')}: {seg.get('text', '')[:60]}...")
        return '\n'.join(lines)

    def _query_terms_present(self, question: str, segments: List[Dict[str, Any]]) -> bool:
        # Check if any main query term is present in the expanded context
        terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
        for seg in segments:
            text = seg.get('text', '').lower()
            if any(term in text for term in terms):
                return True
        return False

    def _find_closest_content(self, question: str) -> str:
        # Return the most similar segment(s) by simple keyword overlap
        terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
        best_score = 0
        best_text = ''
        for seg in self.all_segments:
            text = seg.get('text', '').lower()
            score = sum(1 for term in terms if term in text)
            if score > best_score:
                best_score = score
                best_text = text
        return best_text[:200] + ('...' if best_text else '') 