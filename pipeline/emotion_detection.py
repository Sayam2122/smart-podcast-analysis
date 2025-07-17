import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

from utils.logger import get_logger
from utils.file_utils import get_file_utils

logger = get_logger(__name__)


class EmotionDetection:
    """
    Dual-mode emotion detection using text and audio analysis
    - Text: DistilRoBERTa for text-based emotion classification
    - Audio: Wav2Vec2 or MFCC features for audio-based emotion recognition
    """
    
    def __init__(self, 
                 text_model: str = "j-hartmann/emotion-english-distilroberta-base",
                 audio_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                 device: str = "auto"):
        """
        Initialize emotion detection models
        
        Args:
            text_model: Hugging Face model for text emotion detection
            audio_model: Hugging Face model for audio emotion detection
            device: Device to use ("cpu", "cuda", "auto")
        """
        self.text_model_name = text_model
        self.audio_model_name = audio_model
        self.device = self._determine_device(device)
        
        # Models (loaded lazily)
        self.text_model = None
        self.text_tokenizer = None
        self.audio_model = None
        self.audio_processor = None
        
        # Emotion mappings
        self.text_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.audio_emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        
        self.file_utils = get_file_utils()
        
        logger.info(f"Initializing emotion detection | Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def analyze_segments(self, segments: List[Dict], audio_data: np.ndarray = None, sample_rate: int = 16000, combine_modes: bool = True) -> List[Dict]:
        """
        Analyze segments for emotions using text and optionally audio, combining results if requested.
        Args:
            segments: List of enriched transcript segments (with start/end times, text, speaker, etc.)
            audio_data: Full audio waveform (np.ndarray) for audio-based emotion detection
            sample_rate: Audio sample rate
            combine_modes: Whether to combine text and audio emotion predictions
        Returns:
            Segments with emotion predictions (text_emotion, audio_emotion, combined_emotion)
        """
        # Text emotion detection
        segments = self.detect_text_emotions(segments)
        # Audio emotion detection if audio data provided
        if audio_data is not None:
            segments = self.detect_audio_emotions(audio_data, segments, sample_rate)
        # Combine emotions if both modes used
        if combine_modes and audio_data is not None:
            segments = self.combine_emotions(segments)
        return segments

    def detect_text_emotions(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect emotions from text segments
        Args:
            segments: List of transcript segments with text
        Returns:
            List of segments with text emotion predictions
        """
        logger.info(f"Starting text emotion detection for {len(segments)} segments")
        self._load_text_model()
        start_time = time.time()
        emotion_segments = []
        for segment in segments:
            text = segment.get('text', '').strip()
            if not text:
                # No text, assign neutral
                emotion_result = {
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'all_scores': {'neutral': 0.5}
                }
            else:
                if self.text_model == "fallback":
                    emotion_result = self._fallback_text_emotion(text)
                else:
                    emotion_result = self._predict_text_emotion(text)
            # Add emotion info to segment
            emotion_segment = segment.copy()
            emotion_segment['text_emotion'] = emotion_result
            emotion_segments.append(emotion_segment)
        detection_time = time.time() - start_time
        logger.info(f"Text emotion detection completed in {detection_time:.2f}s")
        return emotion_segments
    
    def _predict_text_emotion(self, text: str) -> Dict:
        """Predict emotion from text using transformer model"""
        try:
            # Tokenize
            inputs = self.text_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get scores
            scores = predictions.cpu().numpy()[0]
            emotion_scores = dict(zip(self.text_emotions, scores))
            
            # Get top emotion
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = float(emotion_scores[top_emotion])
            
            return {
                'emotion': top_emotion,
                'confidence': confidence,
                'all_scores': {k: float(v) for k, v in emotion_scores.items()}
            }
            
        except Exception as e:
            logger.warning(f"Text emotion prediction failed: {e}")
            return self._fallback_text_emotion(text)
    
    def _fallback_text_emotion(self, text: str) -> Dict:
        """Fallback rule-based text emotion detection"""
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        emotion_keywords = {
            'joy': ['happy', 'great', 'amazing', 'wonderful', 'excellent', 'love', 'good', 'best'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'terrible', 'awful', 'worst', 'bad'],
            'sadness': ['sad', 'depressed', 'sorry', 'terrible', 'disappointed', 'hurt'],
            'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious', 'terrified'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocking'],
            'disgust': ['disgusting', 'gross', 'awful', 'terrible', 'horrible']
        }
        
        emotion_scores = {'neutral': 0.3}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = min(0.8, score * 0.2 + 0.2)
        
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[top_emotion]
        
        # Normalize scores
        total = sum(emotion_scores.values())
        normalized_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'all_scores': normalized_scores
        }
    
    def detect_audio_emotions(self, 
                             audio_data: np.ndarray, 
                             segments: List[Dict],
                             sample_rate: int = 16000) -> List[Dict]:
        """
        Detect emotions from audio segments
        
        Args:
            audio_data: Full audio data as numpy array
            segments: List of segments with timing information
            sample_rate: Audio sample rate
            
        Returns:
            List of segments with audio emotion predictions
        """
        logger.info(f"Starting audio emotion detection for {len(segments)} segments")
        self._load_audio_model()
        start_time = time.time()
        emotion_segments = []
        for segment in segments:
            # Extract audio for this segment
            start_sample = int(segment['start_time'] * sample_rate)
            end_sample = int(segment['end_time'] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) < sample_rate * 0.5:  # Less than 0.5 seconds
                # Too short for reliable emotion detection
                emotion_result = {
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'all_scores': {'neutral': 0.5}
                }
            else:
                # Use fallback if model is unreliable or explicitly fallback
                if self.audio_model == "fallback" or not getattr(self, "audio_model_reliable", True):
                    emotion_result = self._fallback_audio_emotion(segment_audio, sample_rate)
                else:
                    emotion_result = self._predict_audio_emotion(segment_audio, sample_rate)
            
            # Add emotion info to segment
            emotion_segment = segment.copy()
            emotion_segment['audio_emotion'] = emotion_result
            emotion_segments.append(emotion_segment)
        
        detection_time = time.time() - start_time
        logger.info(f"Audio emotion detection completed in {detection_time:.2f}s")
        
        return emotion_segments
    
    def _predict_audio_emotion(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Predict emotion from audio using Wav2Vec2 model"""
        try:
            # Resample if necessary (Wav2Vec2 typically expects 16kHz)
            if sample_rate != 16000:
                audio_data = self._resample_audio(audio_data, sample_rate, 16000)

            # Preprocess audio
            inputs = self.audio_processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get scores
            scores = predictions.cpu().numpy()[0]
            model_emotions = self.audio_emotions.copy()
            emotion_scores = dict(zip(model_emotions, scores))

            # Map 'happiness' to 'joy' for consistency
            if 'happiness' in emotion_scores:
                emotion_scores['joy'] = emotion_scores.pop('happiness')

            # Ensure all expected emotions are present
            expected_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            missing = [e for e in expected_emotions if e not in emotion_scores]
            if missing:
                logger.warning(f"Audio model output missing emotions: {missing}. Filling with small values.")
                for e in missing:
                    emotion_scores[e] = 0.01

            # Normalize scores
            total = sum(emotion_scores.values())
            normalized_scores = {k: float(v)/total for k, v in emotion_scores.items()}

            # Get top emotion
            top_emotion = max(normalized_scores, key=normalized_scores.get)
            confidence = float(normalized_scores[top_emotion])

            return {
                'emotion': top_emotion,
                'confidence': confidence,
                'all_scores': normalized_scores
            }

        except Exception as e:
            logger.warning(f"Audio emotion prediction failed: {e}")
            return self._fallback_audio_emotion(audio_data, sample_rate)
    
    def _fallback_audio_emotion(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Fallback MFCC-based audio emotion detection"""
        try:
            import librosa

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)

            # Simple rule-based classification based on MFCC statistics
            energy = np.mean(audio_data ** 2)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))

            # Expanded heuristic emotion classification
            emotion_scores = {
                'anger': 0.05,
                'disgust': 0.05,
                'fear': 0.05,
                'joy': 0.05,
                'neutral': 0.7,
                'sadness': 0.05,
                'surprise': 0.05
            }

            # High energy and spectral centroid often indicate excitement/anger
            if energy > 0.01 and spectral_centroid > 2000:
                emotion_scores['joy'] += 0.25
                emotion_scores['anger'] += 0.15
                emotion_scores['neutral'] -= 0.2
            elif energy < 0.005:
                emotion_scores['sadness'] += 0.25
                emotion_scores['neutral'] -= 0.2
            elif zero_crossing_rate > 0.1:
                emotion_scores['surprise'] += 0.25
                emotion_scores['neutral'] -= 0.2

            # Heuristic for disgust: low energy, high spectral rolloff
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
            if energy < 0.007 and spectral_rolloff > 4000:
                emotion_scores['disgust'] += 0.15
                emotion_scores['neutral'] -= 0.1

            # Heuristic for fear: high zero crossing rate and low energy
            if zero_crossing_rate > 0.12 and energy < 0.008:
                emotion_scores['fear'] += 0.15
                emotion_scores['neutral'] -= 0.1

            # Ensure no negative scores
            for k in emotion_scores:
                if emotion_scores[k] < 0:
                    emotion_scores[k] = 0.01

            # Normalize scores
            total = sum(emotion_scores.values())
            normalized_scores = {k: v/total for k, v in emotion_scores.items()}

            top_emotion = max(normalized_scores, key=normalized_scores.get)
            confidence = normalized_scores[top_emotion]

            return {
                'emotion': top_emotion,
                'confidence': confidence,
                'all_scores': normalized_scores
            }

        except ImportError:
            # If librosa not available, return neutral with all emotions present
            neutral_scores = {
                'anger': 0.05,
                'disgust': 0.05,
                'fear': 0.05,
                'joy': 0.05,
                'neutral': 0.7,
                'sadness': 0.05,
                'surprise': 0.05
            }
            total = sum(neutral_scores.values())
            normalized_scores = {k: v/total for k, v in neutral_scores.items()}
            return {
                'emotion': 'neutral',
                'confidence': normalized_scores['neutral'],
                'all_scores': normalized_scores
            }
        except Exception as e:
            logger.warning(f"Fallback audio emotion detection failed: {e}")
            neutral_scores = {
                'anger': 0.05,
                'disgust': 0.05,
                'fear': 0.05,
                'joy': 0.05,
                'neutral': 0.7,
                'sadness': 0.05,
                'surprise': 0.05
            }
            total = sum(neutral_scores.values())
            normalized_scores = {k: v/total for k, v in neutral_scores.items()}
            return {
                'emotion': 'neutral',
                'confidence': normalized_scores['neutral'],
                'all_scores': normalized_scores
            }
    
    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            return np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
    
    def combine_emotions(self, segments: List[Dict]) -> List[Dict]:
        """
        Combine text and audio emotion predictions
        
        Args:
            segments: Segments with both text and audio emotions
            
        Returns:
            Segments with combined emotion predictions
        """
        logger.info("Combining text and audio emotion predictions")
        
        combined_segments = []
        
        for segment in segments:
            text_emotion = segment.get('text_emotion', {})
            audio_emotion = segment.get('audio_emotion', {})
            
            # Combine emotions using weighted average
            combined_emotion = self._combine_emotion_predictions(text_emotion, audio_emotion)
            
            # Add combined emotion to segment
            combined_segment = segment.copy()
            combined_segment['emotions'] = {
                'text_emotion': text_emotion,
                'audio_emotion': audio_emotion,
                'combined_emotion': combined_emotion
            }
            
            combined_segments.append(combined_segment)
        
        return combined_segments
    
    def _combine_emotion_predictions(self, 
                                   text_emotion: Dict, 
                                   audio_emotion: Dict,
                                   text_weight: float = 0.6,
                                   audio_weight: float = 0.4) -> Dict:
        """Combine text and audio emotion predictions"""
        if not text_emotion and not audio_emotion:
            return {'emotion': 'neutral', 'confidence': 0.5}
        
        if not text_emotion:
            return audio_emotion
        
        if not audio_emotion:
            return text_emotion
        
        # Get all possible emotions
        all_emotions = set(text_emotion.get('all_scores', {}).keys()) | \
                      set(audio_emotion.get('all_scores', {}).keys())
        
        # Combine scores
        combined_scores = {}
        for emotion in all_emotions:
            text_score = text_emotion.get('all_scores', {}).get(emotion, 0.0)
            audio_score = audio_emotion.get('all_scores', {}).get(emotion, 0.0)
            
            combined_score = text_weight * text_score + audio_weight * audio_score
            combined_scores[emotion] = combined_score
        
        # Get top emotion
        top_emotion = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[top_emotion]
        
        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'all_scores': combined_scores,
            'method': 'combined'
        }
    
    def save_emotions(self, segments: List[Dict], output_dir: str) -> None:
        """
        Save emotion detection results
        
        Args:
            segments: Segments with emotion predictions
            output_dir: Directory to save results
        """
        # Save text emotions
        text_emotions = []
        audio_emotions = []
        combined_emotions = []
        
        for segment in segments:
            base_info = {
                'segment_id': segment.get('segment_id'),
                'start_time': segment.get('start_time'),
                'end_time': segment.get('end_time'),
                'text': segment.get('text', '')
            }
            
            # Text emotions
            if 'text_emotion' in segment:
                text_emotion_data = base_info.copy()
                text_emotion_data.update(segment['text_emotion'])
                text_emotions.append(text_emotion_data)
            
            # Audio emotions
            if 'audio_emotion' in segment:
                audio_emotion_data = base_info.copy()
                audio_emotion_data.update(segment['audio_emotion'])
                audio_emotions.append(audio_emotion_data)
            
            # Combined emotions
            if 'emotions' in segment:
                combined_emotion_data = base_info.copy()
                combined_emotion_data.update(segment['emotions']['combined_emotion'])
                combined_emotions.append(combined_emotion_data)
        
        # Save files
        if text_emotions:
            self.file_utils.save_json(
                {'segments': text_emotions},
                f"{output_dir}/emotions_text.json"
            )
        
        if audio_emotions:
            self.file_utils.save_json(
                {'segments': audio_emotions},
                f"{output_dir}/emotions_audio.json"
            )
        
        if combined_emotions:
            self.file_utils.save_json(
                {'segments': combined_emotions},
                f"{output_dir}/emotions_combined.json"
            )
        
        logger.info(f"Saved emotion detection results to {output_dir}")
    
    def get_emotion_stats(self, segments: List[Dict]) -> Dict:
        """
        Get statistics about detected emotions
        
        Args:
            segments: Segments with emotion predictions
            
        Returns:
            Dictionary with emotion statistics
        """
        if not segments:
            return {}
        
        # Extract emotions
        emotions = []
        for segment in segments:
            if 'emotions' in segment and 'combined_emotion' in segment['emotions']:
                emotions.append(segment['emotions']['combined_emotion']['emotion'])
            elif 'text_emotion' in segment:
                emotions.append(segment['text_emotion']['emotion'])
            elif 'audio_emotion' in segment:
                emotions.append(segment['audio_emotion']['emotion'])
        
        if not emotions:
            return {}
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate percentages
        total = len(emotions)
        emotion_percentages = {
            emotion: (count / total) * 100
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'total_segments': total,
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'emotion_diversity': len(emotion_counts)
        }
    
    def _load_text_model(self):
        """Load text emotion detection model"""
        if self.text_model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading text emotion model: {self.text_model_name}")
            
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name)
            
            if self.device == "cuda":
                self.text_model = self.text_model.to("cuda")
            
            self.text_model.eval()
            logger.info("Text emotion model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load text emotion model: {e}")
            logger.info("Falling back to rule-based text emotion detection")
            self.text_model = "fallback"

    def _load_audio_model(self):
        """Load audio emotion detection model"""
        if self.audio_model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForAudioClassification
            import logging
            logger.info(f"Loading audio emotion model: {self.audio_model_name}")

            # Temporarily capture warnings
            class WarningCatcher(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.messages = []
                def emit(self, record):
                    self.messages.append(record.getMessage())

            catcher = WarningCatcher()
            logging.getLogger("transformers.modeling_utils").addHandler(catcher)

            self.audio_processor = AutoProcessor.from_pretrained(self.audio_model_name)
            self.audio_model = AutoModelForAudioClassification.from_pretrained(self.audio_model_name)

            # Remove warning catcher
            logging.getLogger("transformers.modeling_utils").removeHandler(catcher)

            # Check for reliability warning
            unreliable = any("newly initialized" in msg for msg in catcher.messages)
            if unreliable:
                logger.warning("Audio emotion model weights are incomplete; predictions may be unreliable. Falling back to MFCC-based detection if needed.")
                self.audio_model_reliable = False
            else:
                self.audio_model_reliable = True

            if self.device == "cuda":
                self.audio_model = self.audio_model.to("cuda")

            self.audio_model.eval()
            logger.info("Audio emotion model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load audio emotion model: {e}")
            logger.info("Falling back to MFCC-based audio emotion detection")
            self.audio_model = "fallback"
            self.audio_model_reliable = False

def detect_emotions_from_segments(segments: List[Dict],
                                 audio_data: np.ndarray = None,
                                 sample_rate: int = 16000,
                                 combine_modes: bool = True) -> List[Dict]:
    """
    Convenience function to detect emotions from segments
    
    Args:
        segments: Transcript segments
        audio_data: Audio data for audio emotion detection
        sample_rate: Audio sample rate
        combine_modes: Whether to combine text and audio emotions
        
    Returns:
        Segments with emotion predictions
    """
    detector = EmotionDetection()
    
    # Text emotion detection
    segments = detector.detect_text_emotions(segments)
    
    # Audio emotion detection if audio data provided
    if audio_data is not None:
        segments = detector.detect_audio_emotions(audio_data, segments, sample_rate)
    
    # Combine emotions if both modes used
    if combine_modes and audio_data is not None:
        segments = detector.combine_emotions(segments)
    
    return segments
