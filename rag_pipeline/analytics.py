from typing import List, Dict, Any
from collections import Counter, defaultdict

class Analytics:
    """
    Provides advanced analytics for podcast segments: speaker dynamics, emotional patterns,
    content metrics, and topic evolution.
    """
    @staticmethod
    def speaker_dynamics(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze speaker distribution, turns, and interaction patterns.
        """
        speakers = [seg.get('speaker', 'Unknown') for seg in segments]
        speaker_counts = Counter(speakers)
        speaker_turns = sum(1 for i in range(1, len(speakers)) if speakers[i] != speakers[i-1])
        return {
            'speaker_counts': dict(speaker_counts),
            'speaker_turns': speaker_turns,
            'unique_speakers': list(speaker_counts.keys())
        }

    @staticmethod
    def emotional_patterns(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze emotional trends and confidence scores across segments.
        """
        text_emotions = [seg.get('text_emotion', {}).get('emotion', 'neutral') for seg in segments]
        audio_emotions = [seg.get('audio_emotion', {}).get('emotion', 'neutral') for seg in segments]
        text_counts = Counter(text_emotions)
        audio_counts = Counter(audio_emotions)
        return {
            'text_emotion_distribution': dict(text_counts),
            'audio_emotion_distribution': dict(audio_counts)
        }

    @staticmethod
    def content_metrics(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute word count, content density, and average segment length.
        """
        word_counts = [len(seg.get('text', '').split()) for seg in segments]
        densities = [seg.get('block_stats', {}).get('compression_ratio', 1.0) for seg in segments]
        return {
            'total_word_count': sum(word_counts),
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'avg_content_density': sum(densities) / len(densities) if densities else 1.0
        }

    @staticmethod
    def topic_evolution(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Track topic/keyword evolution across segments.
        """
        topic_map = defaultdict(list)
        for seg in segments:
            for topic in seg.get('block_key_points', []):
                topic_map[topic].append(seg.get('segment_id'))
        return {
            'topics': dict(topic_map),
            'topic_counts': {k: len(v) for k, v in topic_map.items()}
        }

    @staticmethod
    def cross_block_emotion(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze emotional patterns across blocks.
        """
        block_emotions = defaultdict(list)
        for seg in segments:
            block_id = seg.get('block_id')
            emotion = seg.get('text_emotion', {}).get('emotion', 'neutral')
            block_emotions[block_id].append(emotion)
        block_summary = {bid: dict(Counter(emotions)) for bid, emotions in block_emotions.items()}
        return block_summary 