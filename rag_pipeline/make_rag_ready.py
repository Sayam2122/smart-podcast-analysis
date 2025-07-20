import json
import os

# Paths
BASE_EPISODES = os.path.join('data', 'episodes')

# Helper to load JSON
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_episode(episode_dir):
    summ_path = os.path.join(episode_dir, 'summarization.json')
    final_path = os.path.join(episode_dir, 'final_report.json')
    emotion_path = os.path.join(episode_dir, 'emotion_detection.json')
    output_path = os.path.join(episode_dir, 'rag_ready.json')

    # Only process if all required files exist and rag_ready.json does not
    if not (os.path.exists(summ_path) and os.path.exists(final_path) and os.path.exists(emotion_path)):
        return
    if os.path.exists(output_path):
        return

    final_report = load_json(final_path)
    summarization = load_json(summ_path)
    emotion_detection = load_json(emotion_path)

    # Build a lookup for audio_emotion all_scores by segment_id
    audio_emotion_lookup = {
        seg['segment_id']: seg['audio_emotion']['all_scores']
        for seg in emotion_detection
        if 'audio_emotion' in seg and 'all_scores' in seg['audio_emotion']
    }

    # Prepare global data
    global_data = {k: v for k, v in final_report.items() if k != 'processing_performance'}

    # Prepare blocks
    blocks = []
    for block in summarization:
        block_copy = dict(block)
        segments = []
        for seg in block_copy.get('segments', []):
            # Get audio_emotion all_scores from emotion_detection.json if available
            audio_scores = audio_emotion_lookup.get(seg['segment_id'], seg['audio_emotion'].get('all_scores', {}))
            # Find the emotion with the highest score
            if audio_scores:
                max_emotion = max(audio_scores, key=audio_scores.get)
            else:
                max_emotion = seg['audio_emotion'].get('emotion', None)
            seg_out = {
                'segment_id': seg['segment_id'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'text': seg['text'],
                'speaker': seg['speaker'],
                'text_emotion': {
                    'emotion': seg['text_emotion']['emotion'],
                    'all_scores': seg['text_emotion']['all_scores']
                },
                'audio_emotion': {
                    'emotion': max_emotion,
                    'all_scores': audio_scores
                }
            }
            segments.append(seg_out)
        block_copy['segments'] = segments
        blocks.append(block_copy)

    rag_ready = {
        'global': global_data,
        'blocks': blocks
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_ready, f, ensure_ascii=False, indent=2)
    print(f"Created: {output_path}")

def main():
    for entry in os.listdir(BASE_EPISODES):
        episode_dir = os.path.join(BASE_EPISODES, entry)
        if os.path.isdir(episode_dir):
            process_episode(episode_dir)

if __name__ == '__main__':
    main() 