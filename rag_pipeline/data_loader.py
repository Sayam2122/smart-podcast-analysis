# /podcast_rag_project/data_loader.py

import os
import json
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

import re

def parse_srt_file(srt_path):
    """
    Parse an SRT file and return a list of segments with start_time, end_time, and text.
    """
    segments = []
    if not os.path.exists(srt_path):
        return segments
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.compile(r'(\d+)\s+([\d:,]+) --> ([\d:,]+)\s+([\s\S]*?)(?=\n\d+\n|\Z)', re.MULTILINE)
    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start = match.group(2).replace(',', '.')
        end = match.group(3).replace(',', '.')
        text = match.group(4).replace('\n', ' ').strip()
        def srt_time_to_sec(t):
            h, m, s = t.split(':')
            s, ms = s.split('.') if '.' in s else (s, '0')
            return int(h)*3600 + int(m)*60 + int(s) + float('0.'+ms)
        segments.append({
            'srt_index': idx,
            'start_time': srt_time_to_sec(start),
            'end_time': srt_time_to_sec(end),
            'text': text
        })
    return segments

class DataLoader:
    """
    Loads and validates podcast episode data from rag_ready.json.
    Builds an episode model: themes, emotional arcs, speaker patterns, key moments.
    Provides access to all segments and the episode model for downstream processing.
    """
    def __init__(self, episodes_dir: str):
        self.episodes_dir = episodes_dir
        self.episode_model = None

    def load_episode(self, episode_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Loads all segments from rag_ready.json for the given episode.
        Builds the episode model for deep podcast understanding.
        Returns a flat list of segments, each with block/segment/source metadata.
        Returns None if file is missing or invalid.
        """
        episode_path = os.path.join(self.episodes_dir, episode_id)
        rag_ready_path = os.path.join(episode_path, 'rag_ready.json')
        if not os.path.exists(rag_ready_path):
            print(f"[DataLoader] rag_ready.json not found for episode: {episode_id}")
            return None
        try:
            with open(rag_ready_path, 'r', encoding='utf-8') as f:
                rag_ready = json.load(f)
            segments = []
            for block in rag_ready.get('blocks', []):
                block_id = block.get('block_id')
                block_summary = block.get('summary')
                block_key_points = block.get('key_points')
                block_insights = block.get('insights')
                block_stats = block.get('summary_stats')
                for seg in block.get('segments', []):
                    enriched = dict(seg)
                    enriched['block_id'] = block_id
                    enriched['block_summary'] = block_summary
                    enriched['block_key_points'] = block_key_points
                    enriched['block_insights'] = block_insights
                    enriched['block_stats'] = block_stats
                    segments.append(enriched)
            self.episode_model = self._build_episode_model(segments)
            return segments
        except Exception as e:
            print(f"[DataLoader] Error loading rag_ready.json for episode {episode_id}: {e}")
            return None

    def get_episode_model(self) -> Optional[Dict[str, Any]]:
        """
        Returns the episode model (themes, emotional arcs, speaker patterns, key moments).
        """
        return self.episode_model

    def _build_episode_model(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze segments/blocks for:
        - Main themes (from block_key_points, block_insights)
        - Emotional arcs (sequence of emotions, peaks, transitions)
        - Speaker patterns (who speaks most, speaker changes)
        - Key moments (emotional peaks, quotable)
        Returns a structured summary.
        """
        # Themes
        all_themes = []
        for seg in segments:
            if seg.get('block_key_points'):
                all_themes.extend(seg['block_key_points'])
            if seg.get('block_insights') and 'theme' in seg['block_insights']:
                all_themes.append(seg['block_insights']['theme'])
        theme_counts = Counter(all_themes)
        # Emotional arcs
        emotions = [(seg.get('start_time', 0), seg.get('text_emotion', {}).get('emotion', 'neutral')) for seg in segments]
        emotion_sequence = [e for _, e in sorted(emotions)]
        emotion_peaks = [seg for seg in segments if max(seg.get('text_emotion', {}).get('all_scores', {}).values() or [0]) > 0.7]
        # Speaker patterns
        speakers = [seg.get('speaker', 'Unknown') for seg in segments]
        speaker_counts = Counter(speakers)
        speaker_turns = sum(1 for i in range(1, len(speakers)) if speakers[i] != speakers[i-1])
        # Key moments (emotional peaks, quotable)
        key_moments = [
            {
                'segment_id': seg.get('segment_id'),
                'block_id': seg.get('block_id'),
                'start_time': seg.get('start_time'),
                'speaker': seg.get('speaker'),
                'emotion': seg.get('text_emotion', {}).get('emotion', 'neutral'),
                'text': seg.get('text', '')
            }
            for seg in segments if max(seg.get('text_emotion', {}).get('all_scores', {}).values() or [0]) > 0.7 or len(seg.get('text', '').split()) > 20
        ]
        return {
            'main_themes': theme_counts.most_common(5),
            'emotion_sequence': emotion_sequence,
            'emotion_peaks': key_moments[:5],
            'speaker_counts': dict(speaker_counts),
            'speaker_turns': speaker_turns,
            'key_moments': key_moments[:10]
        }
