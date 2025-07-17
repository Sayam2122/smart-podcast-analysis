"""
Semantic segmentation module for the podcast analysis pipeline.
Segments transcript into semantic blocks using topic modeling and embeddings.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class SemanticSegmentation:
    """
    Semantic segmentation using sentence embeddings and topic modeling
    Breaks transcript into coherent semantic blocks
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 min_block_size: int = 3,
                 similarity_threshold: float = 0.3,
                 device: str = "auto"):
        """
        Initialize semantic segmentation
        
        Args:
            embedding_model: SentenceTransformers model for embeddings
            min_block_size: Minimum number of segments per block
            similarity_threshold: Similarity threshold for clustering
            device: Device to use
        """
        self.embedding_model_name = embedding_model
        self.min_block_size = min_block_size
        self.similarity_threshold = similarity_threshold
        self.device = self._determine_device(device)
        
        # Models (loaded lazily)
        self.embedding_model = None
        self.topic_model = None
        
        self.file_utils = get_file_utils()
        
        logger.info(f"Initializing semantic segmentation | "
                   f"Model: {embedding_model} | "
                   f"Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except:
                return "cpu"
        return device
    
    def _load_embedding_model(self):
        """Load sentence embedding model"""
        if self.embedding_model is not None:
            return
        
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"Loaded embedding model in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.info("Falling back to simple text similarity")
            self.embedding_model = "fallback"
    
    def _load_topic_model(self):
        """Load topic modeling components"""
        if self.topic_model is not None:
            return
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.cluster import KMeans
            
            self.topic_model = {
                'vectorizer': TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                ),
                'lda': LatentDirichletAllocation(
                    n_components=8,
                    random_state=42,
                    max_iter=10
                ),
                'kmeans': KMeans(n_clusters=8, random_state=42, n_init=10)
            }
            
            logger.info("Loaded topic modeling components")
            
        except Exception as e:
            logger.error(f"Failed to load topic modeling: {e}")
            self.topic_model = "fallback"
    
    def segment_transcript(self, segments: List[Dict]) -> List[Dict]:
        """
        Segment transcript into semantic blocks
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of semantic blocks with grouped segments
        """
        logger.info(f"Starting semantic segmentation for {len(segments)} segments")
        
        if len(segments) < self.min_block_size:
            # Not enough segments for meaningful segmentation
            return self._create_single_block(segments)
        
        start_time = time.time()
        
        # Extract text for analysis
        texts = [seg.get('text', '') for seg in segments]
        
        # Method 1: Embedding-based segmentation
        embedding_blocks = self._segment_by_embeddings(segments, texts)
        
        # Method 2: Topic-based segmentation
        topic_blocks = self._segment_by_topics(segments, texts)
        
        # Method 3: Sliding window approach
        window_blocks = self._segment_by_sliding_window(segments, texts)
        
        # Combine and refine segmentation results
        final_blocks = self._combine_segmentation_methods(
            segments, embedding_blocks, topic_blocks, window_blocks
        )
        
        # Post-process blocks
        processed_blocks = self._post_process_blocks(final_blocks, segments)
        
        segmentation_time = time.time() - start_time
        logger.info(f"Semantic segmentation completed | "
                   f"Blocks: {len(processed_blocks)} | "
                   f"Time: {segmentation_time:.2f}s")
        
        return processed_blocks
    
    def _segment_by_embeddings(self, segments: List[Dict], texts: List[str]) -> List[List[int]]:
        """Segment using sentence embeddings and similarity"""
        self._load_embedding_model()
        
        if self.embedding_model == "fallback":
            return self._fallback_embedding_segmentation(segments, texts)
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            # Find segment boundaries based on similarity drops
            blocks = []
            current_block = [0]
            
            for i in range(1, len(segments)):
                # Compare with previous segments in current block
                similarities = [similarity_matrix[i][j] for j in current_block[-3:]]  # Last 3 segments
                avg_similarity = np.mean(similarities)
                
                if avg_similarity < self.similarity_threshold or len(current_block) > 15:
                    # Start new block
                    if len(current_block) >= self.min_block_size:
                        blocks.append(current_block)
                    current_block = [i]
                else:
                    current_block.append(i)
            
            # Add last block
            if current_block:
                if len(current_block) >= self.min_block_size:
                    blocks.append(current_block)
                elif blocks:
                    # Merge with last block if too small
                    blocks[-1].extend(current_block)
                else:
                    blocks.append(current_block)
            
            return blocks
            
        except Exception as e:
            logger.warning(f"Embedding segmentation failed: {e}")
            return self._fallback_embedding_segmentation(segments, texts)
    
    def _fallback_embedding_segmentation(self, segments: List[Dict], texts: List[str]) -> List[List[int]]:
        """Fallback segmentation using simple text similarity"""
        blocks = []
        current_block = [0]
        
        for i in range(1, len(segments)):
            # Simple word overlap similarity
            current_words = set(texts[i].lower().split())
            prev_words = set(texts[i-1].lower().split())
            
            if current_words and prev_words:
                overlap = len(current_words & prev_words)
                union = len(current_words | prev_words)
                similarity = overlap / union if union > 0 else 0
            else:
                similarity = 0
            
            if similarity < 0.1 or len(current_block) > 10:
                if len(current_block) >= self.min_block_size:
                    blocks.append(current_block)
                current_block = [i]
            else:
                current_block.append(i)
        
        if current_block:
            if len(current_block) >= self.min_block_size:
                blocks.append(current_block)
            elif blocks:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        
        return blocks
    
    def _segment_by_topics(self, segments: List[Dict], texts: List[str]) -> List[List[int]]:
        """Segment using topic modeling"""
        self._load_topic_model()
        
        if self.topic_model == "fallback":
            return self._fallback_topic_segmentation(segments, texts)
        
        try:
            # Clean and prepare text
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Remove empty texts
            valid_indices = [i for i, text in enumerate(cleaned_texts) if len(text.strip()) > 10]
            valid_texts = [cleaned_texts[i] for i in valid_indices]
            
            if len(valid_texts) < self.min_block_size:
                return [list(range(len(segments)))]
            
            # Vectorize
            tfidf_matrix = self.topic_model['vectorizer'].fit_transform(valid_texts)
            
            # Topic modeling
            lda_topics = self.topic_model['lda'].fit_transform(tfidf_matrix)
            
            # Get dominant topic for each segment
            dominant_topics = np.argmax(lda_topics, axis=1)
            
            # Group segments by topic
            topic_groups = defaultdict(list)
            for i, topic in enumerate(dominant_topics):
                original_index = valid_indices[i]
                topic_groups[topic].append(original_index)
            
            # Convert to blocks (maintain temporal order)
            blocks = []
            for topic, indices in topic_groups.items():
                if len(indices) >= self.min_block_size:
                    blocks.append(sorted(indices))
            
            # Sort blocks by first segment index
            blocks.sort(key=lambda x: x[0])
            
            return blocks
            
        except Exception as e:
            logger.warning(f"Topic segmentation failed: {e}")
            return self._fallback_topic_segmentation(segments, texts)
    
    def _fallback_topic_segmentation(self, segments: List[Dict], texts: List[str]) -> List[List[int]]:
        """Fallback topic segmentation using keyword clustering"""
        # Extract keywords and group by similarity
        blocks = []
        current_block = [0]
        
        for i in range(1, len(segments)):
            # Extract keywords (simple approach)
            current_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', texts[i].lower()))
            prev_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', texts[i-1].lower()))
            
            # Calculate keyword overlap
            if current_keywords and prev_keywords:
                overlap = len(current_keywords & prev_keywords)
                similarity = overlap / min(len(current_keywords), len(prev_keywords))
            else:
                similarity = 0
            
            if similarity < 0.2 or len(current_block) > 12:
                if len(current_block) >= self.min_block_size:
                    blocks.append(current_block)
                current_block = [i]
            else:
                current_block.append(i)
        
        if current_block:
            if len(current_block) >= self.min_block_size:
                blocks.append(current_block)
            elif blocks:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        
        return blocks
    
    def _segment_by_sliding_window(self, segments: List[Dict], texts: List[str]) -> List[List[int]]:
        """Segment using sliding window approach"""
        window_size = max(5, len(segments) // 8)  # Adaptive window size
        blocks = []
        
        i = 0
        while i < len(segments):
            end_idx = min(i + window_size, len(segments))
            block = list(range(i, end_idx))
            
            if len(block) >= self.min_block_size:
                blocks.append(block)
            elif blocks:
                # Merge with previous block if too small
                blocks[-1].extend(block)
            else:
                blocks.append(block)
            
            i += window_size - 2  # Overlap of 2 segments
        
        return blocks
    
    def _combine_segmentation_methods(self, 
                                    segments: List[Dict],
                                    embedding_blocks: List[List[int]],
                                    topic_blocks: List[List[int]],
                                    window_blocks: List[List[int]]) -> List[List[int]]:
        """Combine results from different segmentation methods"""
        # For now, prioritize embedding-based segmentation
        # In the future, could implement voting or weighted combination
        
        if embedding_blocks and len(embedding_blocks) > 1:
            return embedding_blocks
        elif topic_blocks and len(topic_blocks) > 1:
            return topic_blocks
        else:
            return window_blocks
    
    def _post_process_blocks(self, blocks: List[List[int]], segments: List[Dict]) -> List[Dict]:
        """Post-process segmentation blocks"""
        processed_blocks = []
        
        for block_id, segment_indices in enumerate(blocks):
            if not segment_indices:
                continue
            
            # Get segments for this block
            block_segments = [segments[i] for i in segment_indices if i < len(segments)]
            
            if not block_segments:
                continue
            
            # Calculate block metadata
            start_time = min(seg['start_time'] for seg in block_segments)
            end_time = max(seg['end_time'] for seg in block_segments)
            total_text = " ".join(seg['text'] for seg in block_segments)
            
            # Extract key topics/themes
            key_topics = self._extract_key_topics(total_text)
            
            # Create block
            block = {
                'block_id': block_id + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'segment_count': len(block_segments),
                'text': total_text,
                'key_topics': key_topics,
                'segments': block_segments
            }
            
            processed_blocks.append(block)
        
        return processed_blocks
    
    def _create_single_block(self, segments: List[Dict]) -> List[Dict]:
        """Create a single block from all segments"""
        if not segments:
            return []
        
        start_time = min(seg['start_time'] for seg in segments)
        end_time = max(seg['end_time'] for seg in segments)
        total_text = " ".join(seg['text'] for seg in segments)
        key_topics = self._extract_key_topics(total_text)
        
        return [{
            'block_id': 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'segment_count': len(segments),
            'text': total_text,
            'key_topics': key_topics,
            'segments': segments
        }]
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words and common filler words
        words = text.split()
        cleaned_words = [
            word for word in words 
            if len(word) > 2 and word.lower() not in {
                'the', 'and', 'but', 'you', 'that', 'this', 'with', 'for',
                'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will',
                'would', 'could', 'should', 'can', 'may', 'might', 'must',
                'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean'
            }
        ]
        
        return ' '.join(cleaned_words)
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics/themes from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Count word frequency
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Filter out common words
        common_words = {
            'that', 'this', 'with', 'from', 'they', 'were', 'been', 'have',
            'said', 'what', 'when', 'where', 'while', 'which', 'these',
            'those', 'then', 'than', 'them', 'here', 'there', 'very'
        }
        
        key_topics = [
            word for word, count in top_words 
            if word not in common_words and count > 1
        ][:5]
        
        return key_topics
    
    def save_semantic_blocks(self, blocks: List[Dict], output_path: str) -> None:
        """
        Save semantic blocks to JSON file
        
        Args:
            blocks: List of semantic blocks
            output_path: Path to save the results
        """
        semantic_data = {
            'metadata': {
                'total_blocks': len(blocks),
                'total_duration': sum(block['duration'] for block in blocks),
                'segmentation_method': 'combined',
                'min_block_size': self.min_block_size
            },
            'blocks': blocks
        }
        
        self.file_utils.save_json(semantic_data, output_path)
        logger.info(f"Saved semantic blocks to: {output_path}")
    
    def get_segmentation_stats(self, blocks: List[Dict]) -> Dict:
        """
        Get statistics about segmentation
        
        Args:
            blocks: List of semantic blocks
            
        Returns:
            Dictionary with segmentation statistics
        """
        if not blocks:
            return {}
        
        block_durations = [block['duration'] for block in blocks]
        segment_counts = [block['segment_count'] for block in blocks]
        
        return {
            'total_blocks': len(blocks),
            'total_duration': sum(block_durations),
            'average_block_duration': np.mean(block_durations),
            'shortest_block': min(block_durations),
            'longest_block': max(block_durations),
            'average_segments_per_block': np.mean(segment_counts),
            'total_segments': sum(segment_counts)
        }


def segment_transcript_semantically(segments: List[Dict],
                                  min_block_size: int = 3,
                                  similarity_threshold: float = 0.3) -> List[Dict]:
    """
    Convenience function to segment transcript semantically
    
    Args:
        segments: List of transcript segments
        min_block_size: Minimum segments per block
        similarity_threshold: Similarity threshold for clustering
        
    Returns:
        List of semantic blocks
    """
    segmenter = SemanticSegmentation(
        min_block_size=min_block_size,
        similarity_threshold=similarity_threshold
    )
    return segmenter.segment_transcript(segments)
