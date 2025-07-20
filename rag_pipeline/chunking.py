import logging
from typing import List, Dict, Any

# --- Configuration for Chunking ---
# Defines the thresholds for creating new chunks.
# - MAX_CHUNK_DURATION: Maximum duration of a chunk in seconds.
# - PAUSE_THRESHOLD: A pause longer than this (in seconds) will trigger a new chunk.
MAX_CHUNK_DURATION = 120  # 2 minutes
PAUSE_THRESHOLD = 2.0     # 2 seconds

def group_segments_into_chunks(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups small, time-stamped text segments into larger, more coherent chunks.

    This strategy aims to create semantically meaningful chunks by grouping
    segments based on speaker continuity and pauses in the conversation. This
    provides richer context for the RAG system compared to using raw, short
    transcription segments.

    A new chunk is created when:
    1. The speaker changes.
    2. A pause between segments exceeds PAUSE_THRESHOLD.
    3. The current chunk's duration exceeds MAX_CHUNK_DURATION.

    Args:
        segments: A list of segment dictionaries, typically from an ASR or
                  diarization process. Each dict must have 'start_time',
                  'end_time', 'text', and 'speaker'.

    Returns:
        A list of larger chunk dictionaries, each containing combined text,
        timing information, and metadata.
    """
    if not segments:
        return []

    logging.info(f"Starting chunking process on {len(segments)} segments...")
    chunks = []
    current_chunk_segments = []
    
    # Sort segments by start time to ensure chronological order
    segments.sort(key=lambda x: x.get('start_time', 0))

    for i, segment in enumerate(segments):
        if not current_chunk_segments:
            # Start of a new chunk
            current_chunk_segments.append(segment)
            continue

        prev_segment = current_chunk_segments[-1]
        
        # --- Chunk Splitting Logic ---
        speaker_changed = segment.get('speaker') != prev_segment.get('speaker')
        
        pause_duration = segment.get('start_time', 0) - prev_segment.get('end_time', 0)
        long_pause = pause_duration > PAUSE_THRESHOLD
        
        current_duration = prev_segment.get('end_time', 0) - current_chunk_segments[0].get('start_time', 0)
        duration_exceeded = current_duration > MAX_CHUNK_DURATION

        if speaker_changed or long_pause or duration_exceeded:
            # Finalize the current chunk
            chunk = _finalize_chunk(current_chunk_segments)
            chunks.append(chunk)
            # Start a new chunk with the current segment
            current_chunk_segments = [segment]
        else:
            # Add segment to the current chunk
            current_chunk_segments.append(segment)

    # Add the last remaining chunk
    if current_chunk_segments:
        chunk = _finalize_chunk(current_chunk_segments)
        chunks.append(chunk)

    logging.info(f"Chunking complete. Created {len(chunks)} chunks.")
    return chunks

def _finalize_chunk(chunk_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Helper function to combine a list of segments into a single chunk dictionary.
    """
    full_text = " ".join(seg.get('text', '').strip() for seg in chunk_segments)
    start_time = chunk_segments[0].get('start_time')
    end_time = chunk_segments[-1].get('end_time')
    
    # For simplicity, we'll take the speaker of the first segment.
    speaker = chunk_segments[0].get('speaker')
    
    # Aggregate metadata from the source segments
    source_segment_ids = [s.get('segment_id') for s in chunk_segments]
    
    # You could add more aggregated metadata here, e.g., dominant emotion
    # by finding the most frequent emotion in the chunk_segments.
    
    return {
        "text": full_text,
        "start_time": start_time,
        "end_time": end_time,
        "speaker": speaker,
        "source_segment_ids": source_segment_ids,
        # Pass through other relevant metadata from the first segment
        "block_id": chunk_segments[0].get('block_id'),
        "text_emotion": chunk_segments[0].get('text_emotion'),
    } 