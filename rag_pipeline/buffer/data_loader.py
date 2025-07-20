# /podcast_rag_project/data_loader.py

import json
import os
import logging

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
    Handles the loading and merging of all specified data files for a single episode.
    It creates a unified, enriched data structure for each transcribed segment, which is
    essential for the vector store and the query engine.
    """

    def __init__(self, episodes_dir, required_files):
        """
        Initializes the DataLoader.

        Args:
            episodes_dir (str): The path to the root directory containing all episode folders.
            required_files (dict): A dictionary from config.py specifying the necessary files.
        """
        self.episodes_dir = episodes_dir
        self.required_files = required_files

    def load_and_merge_data(self, episode_id):
        """
        Loads and merges data for a given episode_id. If rag_ready.json exists, loads from it directly.
        Otherwise, falls back to the old merging logic.
        Returns a list of enriched segment dictionaries, or None if an error occurs.
        """
        logging.info(f"Starting data loading for episode: {episode_id}")
        episode_path = os.path.join(self.episodes_dir, episode_id)
        rag_ready_path = os.path.join(episode_path, 'rag_ready.json')

        if not os.path.isdir(episode_path):
            logging.error(f"Episode directory not found: {episode_path}")
            return None

        if os.path.exists(rag_ready_path):
            try:
                with open(rag_ready_path, 'r', encoding='utf-8') as f:
                    rag_ready = json.load(f)
                # Flatten blocks/segments into a list of enriched segments
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
                logging.info(f"Loaded {len(segments)} segments from rag_ready.json for episode '{episode_id}'.")
                return segments
            except Exception as e:
                logging.error(f"Failed to load rag_ready.json for episode '{episode_id}': {e}")
                return None

        # --- Fallback: old merging logic ---
        try:
            loaded_data = {}
            for key, filename in self.required_files.items():
                file_path = os.path.join(episode_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data[key] = json.load(f)
                    logging.info(f"Successfully loaded {filename}.")
        except FileNotFoundError as e:
            logging.error(f"Data file not found in episode '{episode_id}': {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from a file in episode '{episode_id}': {e}")
            return None

        transcription_srt_path = os.path.join(episode_path, 'transcription.srt')
        summarization_srt_path = os.path.join(episode_path, 'summarization.srt')
        transcription_srt = parse_srt_file(transcription_srt_path)
        summarization_srt = parse_srt_file(summarization_srt_path)

        rich_summary_map = {item['block_id']: item for item in loaded_data.get('summarization', [])}
        segment_to_block_map = self._create_segment_block_map(
            loaded_data.get('semantic', []), rich_summary_map
        )

        base_segments = loaded_data.get('emotion', [])
        if not base_segments:
            base_segments = [
                {
                    'segment_id': s['srt_index'],
                    'start_time': s['start_time'],
                    'end_time': s['end_time'],
                    'text': s['text'],
                    'speaker': None,
                    'confidence': 1.0
                }
                for s in transcription_srt
            ]

        def find_matching_srt(seg, srt_list):
            for srt in srt_list:
                if not (seg['end_time'] < srt['start_time'] or seg['start_time'] > srt['end_time']):
                    return srt
            return None

        enriched_segments = []
        for segment in base_segments:
            segment_id = segment.get('segment_id')
            block_info = segment_to_block_map.get(segment_id, {})
            srt_match = find_matching_srt(segment, transcription_srt)
            srt_text = srt_match['text'] if srt_match else segment.get('text', '')
            summary_srt_match = find_matching_srt(segment, summarization_srt)
            summary_srt_text = summary_srt_match['text'] if summary_srt_match else ''
            enriched_segments.append({
                **segment,
                **block_info,
                'text': srt_text,
                'summary_srt_text': summary_srt_text
            })

        logging.info(f"Successfully merged and enriched data for {len(enriched_segments)} segments in episode '{episode_id}'.")
        return enriched_segments

    def _create_segment_block_map(self, semantic_blocks, rich_summary_map):
        """
        Helper function to create a mapping from each individual segment_id to the
        full context of its parent semantic block.

        Args:
            semantic_data (list): The data from semantic_segmentation.json.
            rich_summary_map (dict): A lookup map created from summarization.json.

        Returns:
            dict: A dictionary where keys are segment_ids and values are dicts
                  containing the parent block's contextual information.
        """
        segment_to_block = {}
        for block in semantic_blocks:
            block_id = block.get('block_id')
            # Get the rich summary info for this block from the map
            summary_info = rich_summary_map.get(block_id, {})
            
            # For every segment within this block, map it to the block's info
            for seg_id in block.get('segment_ids', []):
                segment_to_block[seg_id] = {
                    'block_id': block_id,
                    'block_summary': summary_info.get('summary'),
                    'block_key_points': summary_info.get('key_points'),
                    'block_insights': summary_info.get('insights'),
                    'block_stats': summary_info.get('summary_stats')
                }
        return segment_to_block
