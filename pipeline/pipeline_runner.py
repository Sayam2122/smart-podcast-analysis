"""
Main pipeline runner for the podcast analysis system.
Orchestrates all pipeline modules and manages session-based processing.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

from utils.logger import get_logger, setup_session_logging
from utils.file_utils import get_file_utils

from pipeline.audio_ingestion import AudioIngestion
from pipeline.transcription import Transcription
from pipeline.diarization import SpeakerDiarization
from pipeline.emotion_detection import EmotionDetection
from pipeline.semantic_segmentation import SemanticSegmentation
from pipeline.summarization import PodcastSummarizer

logger = get_logger(__name__)


class PipelineRunner:
    """
    Main pipeline runner that orchestrates all podcast analysis components
    Supports session-based processing with resume capability
    """
    
    def __init__(self,
                 output_dir: str = "output",
                 session_id: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize pipeline runner
        
        Args:
            output_dir: Base output directory
            session_id: Existing session ID to resume, or None for new session
            config: Configuration dictionary for pipeline components
        """
        self.output_dir = Path(output_dir)
        self.session_id = session_id or self._generate_session_id()
        self.session_dir = self.output_dir / "sessions" / f"session_{self.session_id}"
        
        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session-based logging
        setup_session_logging(str(self.session_dir))
        
        # Initialize file utilities
        self.file_utils = get_file_utils()
        
        # Load or create configuration
        self.config = self._load_or_create_config(config)
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Track processing state
        self.state = self._load_session_state()
        
        logger.info(f"Pipeline runner initialized | Session: {self.session_id}")
        logger.info(f"Session directory: {self.session_dir}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
    
    def _load_or_create_config(self, config: Optional[Dict]) -> Dict:
        """Load configuration from file or create default, always merging with defaults"""
        config_path = self.session_dir / "config.json"

        # Default configuration
        default_config = {
            'audio_ingestion': {
                'target_sample_rate': 16000,
                'noise_gate_threshold': 0.01
            },
            'transcription': {
                'model_size': 'medium',
                'backend': 'faster-whisper',
                'device': 'auto',
                'compute_type': 'float16'
            },
            'diarization': {
                'device': 'auto',
                'num_speakers': None,
                'min_speakers': 1,
                'max_speakers': 8
            },
            'emotion_detection': {
                'text_model': 'j-hartmann/emotion-english-distilroberta-base',
                'audio_model': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                'device': 'auto'
            },
            'semantic_segmentation': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'min_block_size': 3,
                'similarity_threshold': 0.3,
                'device': 'auto'
            },
            'summarization': {
                'model_name': 'mistral:7b',
                'ollama_url': 'http://localhost:11434',
                'max_tokens': 300,
                'temperature': 0.3
            }
        }

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    deep_update(d[k], v)
                else:
                    d[k] = v
            return d

        if config_path.exists():
            logger.info("Loading existing session configuration")
            loaded_config = self.file_utils.load_json(str(config_path))
            merged_config = deep_update(default_config.copy(), loaded_config)
            # If user provided config, merge that too
            if config:
                merged_config = deep_update(merged_config, config)
            self.file_utils.save_json(merged_config, str(config_path))
            return merged_config

        # If no config exists, merge with provided config and save
        merged_config = default_config.copy()
        if config:
            merged_config = deep_update(merged_config, config)
        self.file_utils.save_json(merged_config, str(config_path))
        logger.info("Created new session configuration")
        return merged_config
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components")
        
        # Audio ingestion
        self.audio_ingestion = AudioIngestion(
            target_sample_rate=self.config['audio_ingestion']['target_sample_rate'],
            noise_gate_threshold=self.config['audio_ingestion']['noise_gate_threshold']
        )
        
        # Transcription
        self.transcriber = Transcription(
            model_size=self.config['transcription']['model_size'],
            backend=self.config['transcription']['backend'],
            device=self.config['transcription']['device'],
            compute_type=self.config['transcription']['compute_type']
        )
        
        # Speaker diarization
        self.diarizer = SpeakerDiarization(
            device=self.config['diarization']['device'],
            num_speakers=self.config['diarization']['num_speakers'],
            min_speakers=self.config['diarization']['min_speakers'],
            max_speakers=self.config['diarization']['max_speakers']
        )
        
        # Emotion detection
        self.emotion_detector = EmotionDetection(
            text_model=self.config['emotion_detection']['text_model'],
            audio_model=self.config['emotion_detection']['audio_model'],
            device=self.config['emotion_detection']['device']
        )
        
        # Semantic segmentation
        self.semantic_segmenter = SemanticSegmentation(
            embedding_model=self.config['semantic_segmentation']['embedding_model'],
            min_block_size=self.config['semantic_segmentation']['min_block_size'],
            similarity_threshold=self.config['semantic_segmentation']['similarity_threshold'],
            device=self.config['semantic_segmentation']['device']
        )
        
        # Summarization
        self.summarizer = PodcastSummarizer(
            model_name=self.config['summarization']['model_name'],
            ollama_url=self.config['summarization']['ollama_url'],
            max_tokens=self.config['summarization']['max_tokens'],
            temperature=self.config['summarization']['temperature']
        )
        
        logger.info("All pipeline components initialized")
    
    def _load_session_state(self) -> Dict:
        """Load session processing state"""
        state_path = self.session_dir / "state.json"
        
        if state_path.exists():
            state = self.file_utils.load_json(str(state_path))
            logger.info(f"Loaded session state | Last step: {state.get('last_completed_step', 'none')}")
            return state
        
        # Default state
        return {
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'last_completed_step': None,
            'audio_file': None,
            'steps_completed': [],
            'processing_times': {},
            'total_processing_time': 0
        }
    
    def _save_session_state(self):
        """Save current session state"""
        state_path = self.session_dir / "state.json"
        self.state['last_updated'] = datetime.now().isoformat()
        self.file_utils.save_json(self.state, str(state_path))
    
    def process_audio_file(self, audio_file_path: str, resume: bool = True) -> Dict:
        """
        Process audio file through complete pipeline
        
        Args:
            audio_file_path: Path to audio file
            resume: Whether to resume from existing progress
            
        Returns:
            Dictionary with all processing results
        """
        start_time = time.time()
        
        logger.info(f"Starting podcast analysis pipeline")
        logger.info(f"Audio file: {audio_file_path}")
        logger.info(f"Session: {self.session_id}")
        
        # Update state
        self.state['audio_file'] = audio_file_path
        self._save_session_state()
        
        results = {}
        
        try:
            # Step 1: Audio Ingestion
            results['audio_data'] = self._run_step(
                'audio_ingestion',
                lambda: self.audio_ingestion.load_and_normalize_audio(audio_file_path),
                resume=resume
            )
            
            # Step 2: Transcription
            results['transcription'] = self._run_step(
                'transcription',
                lambda: self.transcriber.transcribe_audio(
                    results['audio_data']['audio'],
                    results['audio_data']['sample_rate']
                ),
                resume=resume
            )
            
            # Step 3: Speaker Diarization
            results['diarization'] = self._run_step(
                'diarization',
                lambda: self.diarizer.diarize_audio(
                    results['audio_data']['audio'],
                    results['audio_data']['sample_rate']
                ),
                resume=resume
            )
            
            # Step 4: Combine transcription with speaker info
            # Robustly extract segments for enrichment
            def get_segments(obj):
                if isinstance(obj, dict) and 'segments' in obj:
                    return obj['segments']
                return obj

            transcription_segments = get_segments(results.get('transcription'))
            diarization_segments = get_segments(results.get('diarization'))

            results['enriched_segments'] = self._run_step(
                'segment_enrichment',
                lambda: self._enrich_segments_with_speakers(
                    transcription_segments,
                    diarization_segments
                ),
                resume=resume
            )
            
            # Step 5: Emotion Detection
            results['emotion_analysis'] = self._run_step(
                'emotion_detection',
                lambda: self.emotion_detector.analyze_segments(
                    results['enriched_segments'],
                    results['audio_data']['audio'],
                    results['audio_data']['sample_rate']
                ),
                resume=resume
            )
            
            # Step 6: Semantic Segmentation
            results['semantic_blocks'] = self._run_step(
                'semantic_segmentation',
                lambda: self.semantic_segmenter.segment_transcript(
                    results['emotion_analysis']
                ),
                resume=resume
            )
            
            # Step 7: Summarization
            def detailed_summarization(blocks):
                # Compose a detailed prompt for each block
                prompts = []
                for block in blocks:
                    block_text = block.get('text', '')
                    prompt = (
                        "You are an expert podcast analyst and educator. "
                        "Write a long, detailed, and explainative summary for the following podcast segment, suitable for content creators and students. "
                        "Include key insights, context, and educational value. "
                        "Make it engaging and easy to understand, as if teaching or helping someone create content.\n\n"
                        f"Segment Text: {block_text}\n"
                        "Summary:"
                    )
                    block['prompt'] = prompt
                return self.summarizer.summarize_blocks(blocks)

            results['summaries'] = self._run_step(
                'summarization',
                lambda: detailed_summarization(results['semantic_blocks']),
                resume=True  # Skip if cached
            )
            
            # Generate final report
            results['final_report'] = self._generate_final_report(results)
            
            # Save all results
            self._save_all_results(results)
            
            total_time = time.time() - start_time
            self.state['total_processing_time'] = total_time
            self.state['last_completed_step'] = 'pipeline_complete'
            self._save_session_state()
            
            logger.info(f"Pipeline completed successfully | Total time: {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.state['error'] = str(e)
            self.state['failed_at'] = datetime.now().isoformat()
            self._save_session_state()
            raise
    
    def _run_step(self, step_name: str, step_function, resume: bool = True) -> Dict:
        """
        Run a pipeline step with caching and resume capability
        
        Args:
            step_name: Name of the step
            step_function: Function to execute
            resume: Whether to load cached results if available
            
        Returns:
            Step results
        """
        # Check if step already completed and we want to resume
        if resume and step_name in self.state.get('steps_completed', []):
            cache_path = self.session_dir / f"{step_name}.json"
            if cache_path.exists():
                logger.info(f"Resuming from cached step: {step_name}")
                return self.file_utils.load_json(str(cache_path))
        
        logger.info(f"Executing step: {step_name}")
        step_start = time.time()
        
        try:
            # Execute step
            result = step_function()
            
            # Save result
            cache_path = self.session_dir / f"{step_name}.json"
            self.file_utils.save_json(result, str(cache_path))
            
            # Update state
            step_time = time.time() - step_start
            self.state['processing_times'][step_name] = step_time
            
            if step_name not in self.state.get('steps_completed', []):
                self.state.setdefault('steps_completed', []).append(step_name)
            
            self.state['last_completed_step'] = step_name
            self._save_session_state()
            
            logger.info(f"Step completed: {step_name} | Time: {step_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Step failed: {step_name} | Error: {e}")
            raise
    
    def _enrich_segments_with_speakers(self, 
                                     transcription_segments,
                                     diarization_segments) -> List[Dict]:
        """
        Advanced alignment: splits transcription segments at speaker boundaries for accurate attribution.
        """
        logger.info("Enriching segments with advanced speaker alignment")

        # Accept both dict and list input
        if isinstance(transcription_segments, dict) and 'segments' in transcription_segments:
            transcription_segments = transcription_segments['segments']
        if isinstance(diarization_segments, dict) and 'segments' in diarization_segments:
            diarization_segments = diarization_segments['segments']

        enriched_segments = []
        seg_id = 1
        for trans_seg in transcription_segments:
            t_start = trans_seg['start_time']
            t_end = trans_seg['end_time']
            t_text = trans_seg['text']
            t_other = {k: v for k, v in trans_seg.items() if k not in ['segment_id', 'start_time', 'end_time', 'duration', 'text']}

            # Find all speaker segments overlapping this transcription segment
            overlapping = [d for d in diarization_segments if not (d['end_time'] <= t_start or d['start_time'] >= t_end)]

            if not overlapping:
                # No speaker info, assign Unknown
                enriched_seg = {
                    'segment_id': seg_id,
                    'start_time': t_start,
                    'end_time': t_end,
                    'duration': t_end - t_start,
                    'text': t_text,
                    'speaker': 'Unknown',
                    'speaker_confidence': 0.0,
                    **t_other
                }
                enriched_segments.append(enriched_seg)
                seg_id += 1
                continue

            # If only one speaker segment overlaps, assign directly
            if len(overlapping) == 1:
                dia_seg = overlapping[0]
                enriched_seg = {
                    'segment_id': seg_id,
                    'start_time': t_start,
                    'end_time': t_end,
                    'duration': t_end - t_start,
                    'text': t_text,
                    'speaker': dia_seg.get('speaker', 'Unknown'),
                    'speaker_confidence': dia_seg.get('confidence', 0.0),
                    **t_other
                }
                enriched_segments.append(enriched_seg)
                seg_id += 1
                continue

            # If multiple speaker segments overlap, split at boundaries
            seg_start = t_start
            for i, dia_seg in enumerate(overlapping):
                seg_end = min(t_end, dia_seg['end_time'])
                # Estimate text proportion for this sub-segment
                prop = (seg_end - seg_start) / (t_end - t_start) if t_end > t_start else 1.0
                sub_text_len = int(len(t_text) * prop)
                sub_text = t_text[:sub_text_len] if i == 0 else t_text[:sub_text_len] if i == len(overlapping)-1 else t_text[:sub_text_len]
                t_text = t_text[sub_text_len:]  # Remove assigned text
                enriched_seg = {
                    'segment_id': seg_id,
                    'start_time': seg_start,
                    'end_time': seg_end,
                    'duration': seg_end - seg_start,
                    'text': sub_text,
                    'speaker': dia_seg.get('speaker', 'Unknown'),
                    'speaker_confidence': dia_seg.get('confidence', 0.0),
                    **t_other
                }
                enriched_segments.append(enriched_seg)
                seg_id += 1
                seg_start = seg_end

        logger.info(f"Enriched {len(enriched_segments)} segments with advanced speaker alignment")
        return enriched_segments
    
    def _generate_final_report(self, results: Dict) -> Dict:
        """Generate comprehensive final report"""
        logger.info("Generating final report")
        
        # Basic statistics
        audio_duration = results['audio_data']['duration']
        total_segments = len(results['enriched_segments'])
        total_blocks = len(results['semantic_blocks'])
        
        # Speaker statistics
        speakers = set(seg['speaker'] for seg in results['enriched_segments'])
        speaker_stats = {}
        for speaker in speakers:
            speaker_segments = [seg for seg in results['enriched_segments'] if seg['speaker'] == speaker]
            speaker_time = sum(seg['end_time'] - seg['start_time'] for seg in speaker_segments)
            speaker_stats[speaker] = {
                'segments': len(speaker_segments),
                'total_time': speaker_time,
                'percentage': (speaker_time / audio_duration) * 100 if audio_duration > 0 else 0
            }
        
        all_emotions = []
        for seg in results.get('emotion_analysis', []):
            # Handle dict type
            if isinstance(seg, dict):
                if 'text_emotion' in seg:
                    te = seg['text_emotion']
                    if isinstance(te, dict):
                        emotion = te.get('emotion')
                        if emotion:
                            all_emotions.append(emotion)
                    elif isinstance(te, str):
                        all_emotions.append(te)
                if 'emotions' in seg:
                    ems = seg['emotions']
                    if isinstance(ems, dict):
                        ce = ems.get('combined_emotion', {})
                        if isinstance(ce, dict):
                            emotion = ce.get('emotion')
                            if emotion:
                                all_emotions.append(emotion)
                        elif isinstance(ce, str):
                            all_emotions.append(ce)
                if 'emotion' in seg:
                    emotion_data = seg['emotion']
                    if isinstance(emotion_data, dict) and 'label' in emotion_data:
                        all_emotions.append(emotion_data['label'])
                    elif isinstance(emotion_data, str):
                        all_emotions.append(emotion_data)
            # Handle string type
            elif isinstance(seg, str):
                all_emotions.append(seg)
            # Handle list type (rare, but possible)
            elif isinstance(seg, list):
                for item in seg:
                    if isinstance(item, dict) and 'label' in item:
                        all_emotions.append(item['label'])
                    elif isinstance(item, str):
                        all_emotions.append(item)
        
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Processing statistics
        processing_times = self.state.get('processing_times', {})
        
        # Generate overall summary and highlights from summarizer
        overall_summary = "No summary available"
        key_highlights = []
        
        if results.get('summaries') and len(results['summaries']) > 0:
            first_summary = results['summaries'][0]
            if isinstance(first_summary, dict):
                overall_summary = first_summary.get('summary', "No summary available")
                key_highlights = first_summary.get('key_points', [])
            elif isinstance(first_summary, str):
                overall_summary = first_summary
            else:
                overall_summary = str(first_summary)
        
        # Generate global summary from all blocks
        global_summary = self._generate_global_summary(results)
        global_highlights = self._extract_global_highlights(results)

        report = {
            'session_info': {
                'session_id': self.session_id,
                'audio_file': self.state['audio_file'],
                'processing_date': datetime.now().isoformat(),
                'total_processing_time': self.state.get('total_processing_time', 0)
            },
            'audio_info': {
                'duration': audio_duration,
                'sample_rate': results['audio_data']['sample_rate'],
                'file_size': results['audio_data'].get('file_size', 0)
            },
            'content_analysis': {
                'total_segments': total_segments,
                'semantic_blocks': total_blocks,
                'speakers_detected': len(speakers),
                'speaker_statistics': speaker_stats,
                'emotion_distribution': emotion_counts
            },
            'overall_summary': overall_summary,
            'global_summary': global_summary,
            'key_highlights': key_highlights,
            'global_highlights': global_highlights,
            'key_insights': self._extract_key_insights(results),
            'processing_performance': {
                'step_times': processing_times,
                'total_time': sum(processing_times.values()),
                'segments_per_second': total_segments / sum(processing_times.values()) if processing_times and sum(processing_times.values()) > 0 else 0
            }
        }
        
        return report
    
    def _extract_key_insights(self, results: Dict) -> List[str]:
        """Extract key insights from analysis results"""
        insights = []
        
        # Speaker insights
        speakers = set(seg['speaker'] for seg in results['enriched_segments'])
        if len(speakers) > 1:
            insights.append(f"Multi-speaker conversation with {len(speakers)} participants")
        
        emotions = []
        for seg in results.get('emotion_analysis', []):
            if isinstance(seg, dict):
                if 'text_emotion' in seg:
                    te = seg['text_emotion']
                    if isinstance(te, dict):
                        emotion = te.get('emotion')
                        if emotion:
                            emotions.append(emotion)
                    elif isinstance(te, str):
                        emotions.append(te)
                if 'emotions' in seg:
                    ems = seg['emotions']
                    if isinstance(ems, dict):
                        ce = ems.get('combined_emotion', {})
                        if isinstance(ce, dict):
                            emotion = ce.get('emotion')
                            if emotion:
                                emotions.append(emotion)
                        elif isinstance(ce, str):
                            emotions.append(ce)
                if 'emotion' in seg:
                    emotion_data = seg['emotion']
                    if isinstance(emotion_data, dict) and 'label' in emotion_data:
                        emotions.append(emotion_data['label'])
                    elif isinstance(emotion_data, str):
                        emotions.append(emotion_data)
            elif isinstance(seg, str):
                emotions.append(seg)
            elif isinstance(seg, list):
                for item in seg:
                    if isinstance(item, dict) and 'label' in item:
                        emotions.append(item['label'])
                    elif isinstance(item, str):
                        emotions.append(item)
        
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else None
        if dominant_emotion:
            insights.append(f"Dominant emotion: {dominant_emotion}")
        
        # Content insights
        if results['semantic_blocks']:
            avg_block_duration = sum(block['duration'] for block in results['semantic_blocks']) / len(results['semantic_blocks'])
            insights.append(f"Average topic duration: {avg_block_duration:.1f} seconds")
        
        # Processing insights
        total_time = self.state.get('total_processing_time', 0)
        audio_duration = results['audio_data']['duration']
        if audio_duration > 0 and total_time > 0:
            speed_factor = audio_duration / total_time
            insights.append(f"Processing speed: {speed_factor:.1f}x real-time")
        
        return insights
    
    def _save_all_results(self, results: Dict):
        """Save all results in various formats, robust to all types"""
        logger.info("Saving all results")

        # Save complete results as JSON
        complete_results_path = self.session_dir / "complete_results.json"
        self.file_utils.save_json(results, str(complete_results_path))

        # Save transcription as SRT
        if 'transcription' in results:
            srt_path = self.session_dir / "transcription.srt"
            transcription = results['transcription']
            segments = None
            if isinstance(transcription, dict) and 'segments' in transcription:
                segments = transcription['segments']
            elif isinstance(transcription, list):
                segments = transcription
            if segments and isinstance(segments, list) and len(segments) > 0 and isinstance(segments[0], dict):
                self._save_srt_file(segments, str(srt_path))
            else:
                logger.warning("Transcription segments not found or invalid format; SRT not saved.")

        # Save final report
        if 'final_report' in results:
            report_path = self.session_dir / "final_report.json"
            self.file_utils.save_json(results['final_report'], str(report_path))

        # Save summary text
        if 'summaries' in results:
            summary_path = self.session_dir / "summary.txt"
            summaries = results['summaries']
            if isinstance(summaries, list) and len(summaries) > 0:
                self._save_summary_text(summaries, str(summary_path))
            else:
                logger.warning("Summaries not found or invalid format; summary.txt not saved.")

        logger.info(f"All results saved to: {self.session_dir}")
    
    def _save_srt_file(self, segments: List[Dict], srt_path: str):
        """Save transcription as SRT subtitle file"""
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_srt_time(segment['start_time'])
                end_time = self._format_srt_time(segment['end_time'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _save_summary_text(self, summaries: List[Dict], summary_path: str):
        """Save summaries as readable text file"""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("PODCAST ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for i, block in enumerate(summaries, 1):
                if not isinstance(block, dict):
                    continue  # skip non-dict items
                f.write(f"SEGMENT {i}\n")
                f.write(f"Time: {block.get('start_time', 0):.1f}s - {block.get('end_time', 0):.1f}s\n")
                f.write(f"Duration: {block.get('duration', 0):.1f}s\n")
                f.write(f"Summary: {block.get('summary', 'No summary available')}\n")
                if block.get('key_points'):
                    f.write("\nKey Points:\n")
                    for point in block['key_points']:
                        f.write(f"- {point}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
    
    def get_session_info(self) -> Dict:
        """Get current session information"""
        return {
            'session_id': self.session_id,
            'session_dir': str(self.session_dir),
            'state': self.state,
            'config': self.config
        }
    
    def list_available_results(self) -> List[str]:
        """List available result files in session directory"""
        result_files = []
        for file_path in self.session_dir.glob("*.json"):
            result_files.append(file_path.name)
        return sorted(result_files)
    
    def cleanup_session(self, keep_final_results: bool = True):
        """
        Clean up session temporary files
        
        Args:
            keep_final_results: Whether to keep final results
        """
        if not keep_final_results:
            import shutil
            shutil.rmtree(self.session_dir)
            logger.info(f"Deleted entire session directory: {self.session_dir}")
        else:
            # Remove intermediate cache files but keep final results
            cache_files = [
                "audio_ingestion.json",
                "transcription.json", 
                "diarization.json",
                "segment_enrichment.json",
                "emotion_detection.json",
                "semantic_segmentation.json"
            ]
            
            for cache_file in cache_files:
                cache_path = self.session_dir / cache_file
                if cache_path.exists():
                    cache_path.unlink()
            
            logger.info("Cleaned up intermediate cache files")
    
    def _generate_global_summary(self, results: Dict) -> str:
        """Generate a comprehensive global summary from all content"""
        try:
            # Collect all text from segments
            all_text = []
            for seg in results.get('enriched_segments', []):
                if isinstance(seg, dict) and 'text' in seg:
                    all_text.append(seg['text'])
            
            if not all_text:
                return "No content available for global summary"
            
            # Create a condensed overview
            total_duration = results.get('audio_data', {}).get('duration', 0)
            speakers = set(seg.get('speaker', 'Unknown') for seg in results.get('enriched_segments', []) if isinstance(seg, dict))
            
            summary_parts = [
                f"This is a {total_duration/60:.1f}-minute audio content featuring {len(speakers)} speaker(s)."
            ]
            
            # Add content themes from semantic blocks
            if results.get('semantic_blocks'):
                block_summaries = []
                for block in results['semantic_blocks'][:3]:  # Top 3 blocks
                    if isinstance(block, dict) and 'summary' in block:
                        block_summaries.append(block['summary'])
                
                if block_summaries:
                    summary_parts.append("Main themes include: " + "; ".join(block_summaries))
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Error generating global summary: {e}")
            return "Global summary generation failed"
    
    def _extract_global_highlights(self, results: Dict) -> List[str]:
        """Extract key highlights from the entire content"""
        highlights = []
        
        try:
            # Speaker highlights
            speakers = set(seg.get('speaker', 'Unknown') for seg in results.get('enriched_segments', []) if isinstance(seg, dict))
            if len(speakers) > 1:
                highlights.append(f"Multi-speaker conversation with {len(speakers)} participants")
            
            # Emotion highlights
            all_emotions = []
            for seg in results.get('emotion_analysis', []):
                if isinstance(seg, dict):
                    # Handle different emotion data structures
                    if 'text_emotion' in seg and isinstance(seg['text_emotion'], dict):
                        emotion = seg['text_emotion'].get('emotion')
                        if emotion:
                            all_emotions.append(emotion)
                    elif 'emotions' in seg and isinstance(seg['emotions'], dict):
                        combined_emotion = seg['emotions'].get('combined_emotion', {})
                        if isinstance(combined_emotion, dict):
                            emotion = combined_emotion.get('emotion')
                            if emotion:
                                all_emotions.append(emotion)
            
            emotion_counts = {}
            for emotion in all_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if emotion_counts:
                top_emotion = max(emotion_counts, key=emotion_counts.get)
                highlights.append(f"Predominant emotional tone: {top_emotion}")
            
            # Content highlights from summaries
            if results.get('summaries'):
                for summary in results['summaries'][:2]:  # Top 2 summaries
                    if isinstance(summary, dict) and 'key_points' in summary:
                        highlights.extend(summary['key_points'][:2])  # Top 2 points each
            
            # Duration highlight
            duration = results.get('audio_data', {}).get('duration', 0)
            if duration > 0:
                highlights.append(f"Total duration: {duration/60:.1f} minutes")
            
            return highlights[:5]  # Return top 5 highlights
            
        except Exception as e:
            logger.warning(f"Error extracting global highlights: {e}")
            return ["Highlights extraction failed"]


def process_podcast(audio_file_path: str,
                   output_dir: str = "output",
                   session_id: Optional[str] = None,
                   resume: bool = True,
                   config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to process a podcast file
    
    Args:
        audio_file_path: Path to audio file
        output_dir: Output directory
        session_id: Session ID to resume (optional)
        resume: Whether to resume from existing progress
        config: Custom configuration
        
    Returns:
        Processing results
    """
    runner = PipelineRunner(
        output_dir=output_dir,
        session_id=session_id,
        config=config
    )
    
    return runner.process_audio_file(audio_file_path, resume=resume)


def list_sessions(output_dir: str = "output") -> List[Dict]:
    """
    List all available sessions
    
    Args:
        output_dir: Output directory to search
        
    Returns:
        List of session information
    """
    sessions_dir = Path(output_dir) / "sessions"
    sessions = []
    
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                state_path = session_dir / "state.json"
                if state_path.exists():
                    try:
                        file_utils = get_file_utils()
                        state = file_utils.load_json(str(state_path))
                        sessions.append({
                            'session_id': state.get('session_id'),
                            'audio_file': state.get('audio_file'),
                            'created_at': state.get('created_at'),
                            'last_completed_step': state.get('last_completed_step'),
                            'session_dir': str(session_dir)
                        })
                    except Exception:
                        continue
    
    return sorted(sessions, key=lambda x: x.get('created_at', ''), reverse=True)
  

# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Podcast Analysis Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file to process')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--session_id', type=str, default=None, help='Session ID to resume or create')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress if available')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config JSON file')
    args = parser.parse_args()

    # Normalize and resolve all paths
    audio_path = str(Path(args.audio_path).expanduser().resolve())
    output_dir = str(Path(args.output_dir).expanduser().resolve())
    config_path = str(Path(args.config).expanduser().resolve()) if args.config else None

    # Check audio file exists
    if not os.path.isfile(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        exit(1)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load custom config if provided
    config = None
    if config_path:
        if not os.path.isfile(config_path):
            print(f"❌ Config file not found: {config_path}")
            exit(1)
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")

    print(f"\n🎙️ Podcast Analysis Pipeline Runner")
    print(f"Audio file: {audio_path}")
    print(f"Session ID: {args.session_id or '[auto]'}")
    print(f"Output directory: {output_dir}")
    print(f"Resume: {args.resume}")
    print("-" * 50)

    try:
        results = process_podcast(
            audio_file_path=audio_path,
            output_dir=output_dir,
            session_id=args.session_id,
            resume=args.resume,
            config=config
        )
        session_id = results['final_report']['session_info']['session_id']
        session_dir = Path(output_dir) / "sessions" / f"session_{session_id}"
        print("\n✅ Pipeline completed successfully!")
        print(f"Session ID: {session_id}")
        print(f"Total processing time: {results['final_report']['session_info']['total_processing_time']:.2f}s")
        print(f"Results saved in: {session_dir.resolve()}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
