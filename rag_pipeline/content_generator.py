from typing import List, Dict, Any
import random

class ContentGenerator:
    """
    Generates social media posts, quote cards, threads, and context-aware content with full source attribution.
    """
    @staticmethod
    def social_media_posts(segments: List[Dict[str, Any]], topic: str = "") -> List[Dict[str, Any]]:
        """
        Generate multi-platform social media posts with hashtags and source info.
        """
        posts = []
        hashtags = ContentGenerator._extract_hashtags(topic, segments)
        for seg in random.sample(segments, min(3, len(segments))):
            post = {
                'platform': 'Twitter',
                'text': f"{seg.get('text', '')[:200]}... #podcast {hashtags}",
                'timestamp': f"{seg.get('start_time', 0):.1f}s",
                'speaker': seg.get('speaker', 'Unknown'),
                'emotion': seg.get('text_emotion', {}).get('emotion', 'neutral'),
                'source': f"Block {seg.get('block_id')}, Segment {seg.get('segment_id')}"
            }
            posts.append(post)
        return posts

    @staticmethod
    def quote_cards(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate visual-ready quote cards with emotional context and source info.
        """
        cards = []
        for seg in random.sample(segments, min(2, len(segments))):
            card = {
                'quote': seg.get('text', '')[:120] + '...',
                'speaker': seg.get('speaker', 'Unknown'),
                'emotion': seg.get('text_emotion', {}).get('emotion', 'neutral'),
                'timestamp': f"{seg.get('start_time', 0):.1f}s",
                'source': f"Block {seg.get('block_id')}, Segment {seg.get('segment_id')}"
            }
            cards.append(card)
        return cards

    @staticmethod
    def thread_generation(segments: List[Dict[str, Any]], topic: str = "") -> List[Dict[str, Any]]:
        """
        Generate a multi-post thread for social platforms, with citations.
        """
        thread = []
        for i, seg in enumerate(random.sample(segments, min(4, len(segments))), 1):
            post = {
                'post_number': i,
                'text': f"[{seg.get('speaker', 'Unknown')} @ {seg.get('start_time', 0):.1f}s] {seg.get('text', '')[:180]}...",
                'emotion': seg.get('text_emotion', {}).get('emotion', 'neutral'),
                'source': f"Block {seg.get('block_id')}, Segment {seg.get('segment_id')}"
            }
            thread.append(post)
        return thread

    @staticmethod
    def _extract_hashtags(topic: str, segments: List[Dict[str, Any]]) -> str:
        """
        Extract or generate relevant hashtags from topic and segment keywords.
        """
        keywords = set()
        if topic:
            keywords.update(topic.lower().split())
        for seg in segments:
            for kp in seg.get('block_key_points', []):
                keywords.add(kp.lower())
        hashtags = [f"#{k.replace(' ', '')}" for k in list(keywords)[:4]]
        return ' '.join(hashtags) 