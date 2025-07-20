# /podcast_rag_project/enhanced_chunking.py

import logging
from typing import List, Dict, Any
import numpy as np

# --- Enhanced Configuration for Contextual Chunking ---
MIN_CHUNK_WORDS = 50        # Minimum words to form a meaningful chunk
MAX_CHUNK_WORDS = 300       # Maximum words before forcing a split
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Threshold for semantic coherence
PAUSE_THRESHOLD = 3.0       # Longer pause threshold for better context
SPEAKER_CONTINUITY_BONUS = 0.2  # Bonus for keeping same speaker together

def enhanced_contextual_chunking(segments: List[Dict[str, Any]], embedding_model=None) -> List[Dict[str, Any]]:
    """
    Advanced contextual chunking that creates semantically meaningful chunks
    by considering semantic coherence, speaker continuity, emotional flow,
    and informational value.
    
    Args:
        segments: List of enriched segment dictionaries
        embedding_model: Optional sentence transformer model for semantic analysis
        
    Returns:
        List of enhanced chunk dictionaries with rich metadata
    """
    if not segments:
        return []
    
    logging.info(f"Starting enhanced contextual chunking on {len(segments)} segments...")
    
    # Sort segments by start time
    segments.sort(key=lambda x: x.get('start_time', 0))
    
    chunks = []
    current_chunk_segments = []
    
    for i, segment in enumerate(segments):
        # Skip segments that are too short or uninformative
        if not _is_segment_informative(segment):
            continue
            
        if not current_chunk_segments:
            current_chunk_segments.append(segment)
            continue
        
        # Analyze whether to continue current chunk or start new one
        should_split = _should_split_chunk(
            current_chunk_segments, 
            segment, 
            embedding_model
        )
        
        if should_split:
            # Finalize current chunk if it meets minimum requirements
            if _meets_chunk_requirements(current_chunk_segments):
                chunk = _create_enhanced_chunk(current_chunk_segments)
                chunks.append(chunk)
            
            # Start new chunk
            current_chunk_segments = [segment]
        else:
            current_chunk_segments.append(segment)
            
            # Force split if chunk becomes too large
            if _is_chunk_too_large(current_chunk_segments):
                chunk = _create_enhanced_chunk(current_chunk_segments)
                chunks.append(chunk)
                current_chunk_segments = []
    
    # Handle remaining segments
    if current_chunk_segments and _meets_chunk_requirements(current_chunk_segments):
        chunk = _create_enhanced_chunk(current_chunk_segments)
        chunks.append(chunk)
    
    logging.info(f"Enhanced chunking complete. Created {len(chunks)} contextual chunks.")
    return chunks

def _is_segment_informative(segment: Dict[str, Any]) -> bool:
    """
    Determines if a segment contains meaningful information worth including.
    
    Args:
        segment: Segment dictionary
        
    Returns:
        Boolean indicating if segment is informative
    """
    text = segment.get('text', '').strip()
    
    # Skip very short segments
    if len(text.split()) < 3:
        return False
    
    # Skip segments that are mostly filler words
    filler_words = {'um', 'uh', 'like', 'you know', 'so', 'well', 'actually'}
    words = text.lower().split()
    filler_ratio = sum(1 for word in words if word in filler_words) / len(words)
    
    if filler_ratio > 0.5:
        return False
    
    # Skip segments with very low confidence
    confidence = segment.get('confidence', 1.0)
    if confidence < 0.3:
        return False
    
    return True

def _should_split_chunk(current_segments: List[Dict[str, Any]], 
                       new_segment: Dict[str, Any], 
                       embedding_model=None) -> bool:
    """
    Determines whether to split the current chunk or continue building it.
    
    Args:
        current_segments: Segments in current chunk
        new_segment: Potential new segment to add
        embedding_model: Optional embedding model for semantic analysis
        
    Returns:
        Boolean indicating whether to split
    """
    if not current_segments:
        return False
    
    last_segment = current_segments[-1]
    
    # Check speaker continuity
    speaker_changed = new_segment.get('speaker') != last_segment.get('speaker')
    
    # Check pause duration
    pause_duration = new_segment.get('start_time', 0) - last_segment.get('end_time', 0)
    long_pause = pause_duration > PAUSE_THRESHOLD
    
    # Check emotional coherence
    emotion_changed = _significant_emotion_change(last_segment, new_segment)
    
    # Check semantic coherence if embedding model available
    semantic_break = False
    if embedding_model:
        semantic_break = _detect_semantic_break(current_segments, new_segment, embedding_model)
    
    # Calculate split score based on multiple factors
    split_score = 0
    
    if speaker_changed:
        split_score += 0.3
    if long_pause:
        split_score += 0.4
    if emotion_changed:
        split_score += 0.2
    if semantic_break:
        split_score += 0.5
    
    # Topic transition indicators in text
    transition_phrases = [
        'moving on', 'next topic', 'another thing', 'switching gears',
        'now let\'s talk about', 'on another note', 'changing subjects'
    ]
    
    new_text = new_segment.get('text', '').lower()
    if any(phrase in new_text for phrase in transition_phrases):
        split_score += 0.3
    
    return split_score > 0.5

def _significant_emotion_change(seg1: Dict[str, Any], seg2: Dict[str, Any]) -> bool:
    """Check if there's a significant change in emotional tone."""
    emotion1 = seg1.get('text_emotion', {}).get('emotion', 'neutral')
    emotion2 = seg2.get('text_emotion', {}).get('emotion', 'neutral')
    
    # Define emotional distance
    emotion_groups = {
        'positive': ['joy', 'surprise'],
        'negative': ['anger', 'sadness', 'fear', 'disgust'],
        'neutral': ['neutral']
    }
    
    group1 = next((g for g, emotions in emotion_groups.items() if emotion1 in emotions), 'neutral')
    group2 = next((g for g, emotions in emotion_groups.items() if emotion2 in emotions), 'neutral')
    
    return group1 != group2

def _detect_semantic_break(current_segments: List[Dict[str, Any]], 
                          new_segment: Dict[str, Any], 
                          embedding_model) -> bool:
    """
    Detect semantic breaks using embedding similarity.
    
    Args:
        current_segments: Current chunk segments
        new_segment: New segment to evaluate
        embedding_model: Sentence transformer model
        
    Returns:
        Boolean indicating semantic break
    """
    try:
        # Get text from current chunk and new segment
        current_text = " ".join([seg.get('text', '') for seg in current_segments[-3:]])  # Last 3 segments
        new_text = new_segment.get('text', '')
        
        # Calculate embeddings
        embeddings = embedding_model.encode([current_text, new_text])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return similarity < SEMANTIC_SIMILARITY_THRESHOLD
        
    except Exception as e:
        logging.warning(f"Could not calculate semantic similarity: {e}")
        return False

def _meets_chunk_requirements(segments: List[Dict[str, Any]]) -> bool:
    """Check if segments meet minimum requirements for a chunk."""
    if not segments:
        return False
    
    total_words = sum(len(seg.get('text', '').split()) for seg in segments)
    return total_words >= MIN_CHUNK_WORDS

def _is_chunk_too_large(segments: List[Dict[str, Any]]) -> bool:
    """Check if chunk exceeds maximum size."""
    total_words = sum(len(seg.get('text', '').split()) for seg in segments)
    return total_words > MAX_CHUNK_WORDS

def _create_enhanced_chunk(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create an enhanced chunk with rich metadata and analysis.
    
    Args:
        segments: List of segments to combine
        
    Returns:
        Enhanced chunk dictionary
    """
    if not segments:
        return {}
    
    # Combine text
    full_text = " ".join(seg.get('text', '').strip() for seg in segments)
    
    # Time boundaries
    start_time = segments[0].get('start_time', 0)
    end_time = segments[-1].get('end_time', 0)
    duration = end_time - start_time
    
    # Speaker analysis
    speakers = [seg.get('speaker') for seg in segments]
    primary_speaker = max(set(speakers), key=speakers.count) if speakers else 'Unknown'
    speaker_changes = len(set(speakers)) - 1
    
    # Emotion analysis
    emotions = [seg.get('text_emotion', {}).get('emotion', 'neutral') for seg in segments]
    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else 'neutral'
    
    # Confidence analysis
    confidences = [seg.get('confidence', 0) for seg in segments if seg.get('confidence')]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Content quality indicators
    word_count = len(full_text.split())
    avg_words_per_segment = word_count / len(segments)
    
    # Create enhanced chunk
    chunk = {
        'text': full_text,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'word_count': word_count,
        'segment_count': len(segments),
        'source_segment_ids': [seg.get('segment_id') for seg in segments],
        
        # Speaker information
        'primary_speaker': primary_speaker,
        'speaker_changes': speaker_changes,
        'all_speakers': list(set(speakers)),
        
        # Emotional information
        'dominant_emotion': dominant_emotion,
        'emotion_distribution': {emotion: emotions.count(emotion) for emotion in set(emotions)},
        
        # Quality metrics
        'avg_confidence': avg_confidence,
        'avg_words_per_segment': avg_words_per_segment,
        'content_density': _calculate_content_density(full_text),
        
        # Preserve important metadata from first segment
        'block_id': segments[0].get('block_id'),
        'block_summary': segments[0].get('block_summary'),
        'block_key_points': segments[0].get('block_key_points', []),
        'text_emotion': segments[0].get('text_emotion', {}),
        
        # Chunk-specific metadata
        'chunk_type': _classify_chunk_type(full_text, emotions),
        'information_value': _assess_information_value(segments),
    }
    
    return chunk

def _calculate_content_density(text: str) -> float:
    """Calculate information density of text content."""
    words = text.split()
    if not words:
        return 0.0
    
    # Simple heuristic: ratio of content words to total words
    content_words = [w for w in words if len(w) > 3 and w.lower() not in {
        'that', 'this', 'with', 'from', 'they', 'were', 'been', 'have', 'will'
    }]
    
    return len(content_words) / len(words)

def _classify_chunk_type(text: str, emotions: List[str]) -> str:
    """Classify the type/nature of the chunk content."""
    text_lower = text.lower()
    
    # Question patterns
    if '?' in text or any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where']):
        return 'question'
    
    # Story/narrative patterns
    if any(word in text_lower for word in ['story', 'happened', 'remember', 'once']):
        return 'narrative'
    
    # Explanation patterns
    if any(word in text_lower for word in ['because', 'therefore', 'explain', 'reason']):
        return 'explanation'
    
    # Emotional content
    emotional_emotions = ['joy', 'anger', 'sadness', 'surprise']
    if any(emotion in emotional_emotions for emotion in emotions):
        return 'emotional'
    
    return 'informational'

def _assess_information_value(segments: List[Dict[str, Any]]) -> float:
    """Assess the information value of the segments."""
    if not segments:
        return 0.0
    
    total_score = 0
    for segment in segments:
        score = 0.5  # Base score
        
        # Confidence bonus
        confidence = segment.get('confidence', 0)
        score += confidence * 0.3
        
        # Length bonus (longer segments often more informative)
        word_count = len(segment.get('text', '').split())
        if word_count > 10:
            score += 0.2
        
        # Emotional content bonus (emotional content often memorable)
        emotion = segment.get('text_emotion', {}).get('emotion', 'neutral')
        if emotion != 'neutral':
            score += 0.1
        
        total_score += min(score, 1.0)  # Cap at 1.0
    
    return total_score / len(segments)
