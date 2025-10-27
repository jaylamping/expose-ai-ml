"""
Fast screening model for bot detection using lightweight transformers.
"""
import torch
import warnings
import numpy as np
import os
import logging
import traceback
from typing import List, Dict, Optional, Union
import time

# Configure logging for this module
logger = logging.getLogger(__name__)

# Set environment variable to suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress all transformers warnings before importing
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*were not used.*")
warnings.filterwarnings("ignore", message=".*This IS expected if you are initializing.*")
warnings.filterwarnings("ignore", message=".*sentencepiece tokenizer.*byte fallback.*")
warnings.filterwarnings("ignore", message=".*roberta.pooler.*")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config.settings import settings
from utils.preprocessing import RedditPreprocessor
from core.device_manager import DeviceManager


class FastDetector:
    """Fast screening model for initial bot detection."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the fast detector.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
        """
        self.model_name = model_name or settings.fast_model_name
        self.device_manager = DeviceManager(device)
        self.device = str(self.device_manager.get_device())
        self.preprocessor = RedditPreprocessor(max_length=settings.max_sequence_length)
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Force CPU on macOS to avoid MPS bus errors
            import platform
            if platform.system() == "Darwin":  # macOS
                device_id = -1  # Force CPU on macOS
                print("macOS detected - forcing CPU usage to avoid MPS bus errors")
            elif "cuda" in self.device:
                device_id = 0  # Use GPU on PC
            else:
                device_id = -1  # Use CPU as fallback
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device_id,
                top_k=None
            )
            print(f"Loaded fast detector pipeline: {self.model_name} on device {device_id}")
        except Exception as e:
            print(f"Pipeline loading failed, trying manual loading: {e}")
            try:
                # Manual loading as fallback - force CPU on macOS
                import platform
                if platform.system() == "Darwin":  # macOS
                    print("macOS detected - forcing CPU usage for manual loading")
                    self.device = "cpu"
                    self.device_manager = DeviceManager("cpu")
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model = self.device_manager.to_device(self.model)
                self.model.eval()
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"Loaded fast detector manually: {self.model_name} on {self.device}")
            except Exception as e2:
                print(f"Failed to load fast detector: {e2}")
                raise e2
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """
        Predict bot probability for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        if not text.strip():
            return {"bot_probability": 0.0, "confidence": 0.0}
        
        # Clean the text
        cleaned_text = self.preprocessor.clean_reddit_text(text)
        
        try:
            if self.pipeline:
                # Use pipeline
                results = self.pipeline(cleaned_text)
                
                # Extract bot probability
                bot_prob = 0.0
                for result in results[0]:
                    if "bot" in result['label'].lower() or "ai" in result['label'].lower() or "fake" in result['label'].lower():
                        bot_prob = result['score']
                        break
                    elif "human" in result['label'].lower() or "real" in result['label'].lower():
                        bot_prob = 1.0 - result['score']
                        break
                
                # If no clear bot/human labels, use the highest score
                if bot_prob == 0.0:
                    bot_prob = max(results[0], key=lambda x: x['score'])['score']
                
                confidence = max(result['score'] for result in results[0])
                
            else:
                # Use manual model
                inputs = self.tokenizer(
                    cleaned_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=settings.max_sequence_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    bot_prob = probabilities[0][1].item()  # Assuming class 1 is bot
                    confidence = torch.max(probabilities).item()
            
            return {
                "bot_probability": bot_prob,
                "confidence": confidence,
                "model": self.model_name
            }
            
        except Exception as e:
            print(f"Error in fast detection: {e}")
            return {"bot_probability": 0.0, "confidence": 0.0, "error": str(e)}
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict bot probability for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction results
        """
        logger.debug(f"FastDetector: predict_batch called with {len(texts)} texts")
        
        if not texts:
            logger.debug("FastDetector: No texts provided, returning empty results")
            return []
        
        try:
            # Clean texts
            logger.debug("FastDetector: Preprocessing texts...")
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            logger.debug(f"FastDetector: Preprocessed {len(cleaned_texts)} texts")
            
            results = []
            
            if self.pipeline:
                logger.debug("FastDetector: Using pipeline for batch processing...")
                # Batch processing with pipeline
                batch_results = self.pipeline(cleaned_texts)
                logger.debug(f"FastDetector: Pipeline returned {len(batch_results)} results")
                
                for i, text_results in enumerate(batch_results):
                    bot_prob = 0.0
                    confidence = 0.0
                    
                    for result in text_results:
                        if "bot" in result['label'].lower() or "ai" in result['label'].lower() or "fake" in result['label'].lower():
                            bot_prob = result['score']
                            break
                        elif "human" in result['label'].lower() or "real" in result['label'].lower():
                            bot_prob = 1.0 - result['score']
                            break
                    
                    if bot_prob == 0.0:
                        bot_prob = max(text_results, key=lambda x: x['score'])['score']
                    
                    confidence = max(result['score'] for result in text_results)
                    
                    results.append({
                        "bot_probability": bot_prob,
                        "confidence": confidence,
                        "model": self.model_name
                    })
            
            else:
                logger.debug("FastDetector: Using manual batch processing...")
                # Manual batch processing
                inputs = self.tokenizer(
                    cleaned_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=settings.max_sequence_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logger.debug(f"FastDetector: Tokenized inputs, moving to device: {self.device}")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    logger.debug(f"FastDetector: Model inference completed, processing {len(texts)} results")
                    
                    for i in range(len(texts)):
                        bot_prob = probabilities[i][1].item()  # Assuming class 1 is bot
                        confidence = torch.max(probabilities[i]).item()
                        
                        results.append({
                            "bot_probability": bot_prob,
                            "confidence": confidence,
                            "model": self.model_name
                        })
            
            logger.debug(f"FastDetector: predict_batch completed successfully with {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"FastDetector: Error in predict_batch: {str(e)}")
            logger.error(f"FastDetector: predict_batch traceback: {traceback.format_exc()}")
            # Return default results for all texts
            results = [{"bot_probability": 0.0, "confidence": 0.0, "error": str(e)} for _ in texts]
            return results
    
    def analyze_user_comments(self, comments: List[str], use_sampling: bool = True) -> Dict[str, Union[float, List, Dict]]:
        """
        Analyze a user's comments with fast screening.
        
        Args:
            comments: List of user comments
            use_sampling: Whether to use smart sampling for large comment sets
            
        Returns:
            Analysis results
        """
        logger.debug(f"FastDetector: Starting analysis of {len(comments)} comments")
        start_time = time.time()
        
        try:
            # Limit comments if too many
            if len(comments) > settings.max_comments_per_request:
                logger.debug(f"FastDetector: Limiting comments from {len(comments)} to {settings.max_comments_per_request}")
                comments = comments[:settings.max_comments_per_request]
            
            # Use smart sampling if requested and we have many comments
            if use_sampling and len(comments) > 25:
                logger.debug(f"FastDetector: Using smart sampling to reduce {len(comments)} comments to 25")
                sampled_comments = self.preprocessor.smart_sampling(comments, sample_size=25)
            else:
                logger.debug(f"FastDetector: Using all {len(comments)} comments")
                sampled_comments = comments
            
            # Get predictions
            logger.debug("FastDetector: Running batch predictions...")
            predictions = self.predict_batch(sampled_comments)
            logger.debug(f"FastDetector: Got {len(predictions)} predictions")
            
            # Extract bot probabilities
            bot_probs = [pred["bot_probability"] for pred in predictions]
            confidences = [pred["confidence"] for pred in predictions]
            
            # Calculate aggregate scores
            mean_bot_prob = np.mean(bot_probs) if bot_probs else 0.0
            mean_confidence = np.mean(confidences) if confidences else 0.0
            std_bot_prob = np.std(bot_probs) if len(bot_probs) > 1 else 0.0
            
            # Determine if we should skip deeper analysis
            should_skip_deep = (
                mean_bot_prob < settings.fast_threshold_low or 
                mean_bot_prob > settings.fast_threshold_high
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.debug(f"FastDetector: Analysis completed in {processing_time:.1f}ms - Bot score: {mean_bot_prob:.3f}, Confidence: {mean_confidence:.3f}")
            
            return {
                "bot_score": mean_bot_prob * 100,  # Convert to percentage
                "confidence": mean_confidence * 100,
                "should_skip_deep": should_skip_deep,
                "processing_time_ms": processing_time,
                "comments_analyzed": len(sampled_comments),
                "total_comments": len(comments),
                "individual_scores": bot_probs,
                "score_std": std_bot_prob,
                "model": self.model_name,
                "stage": "fast_screening"
            }
            
        except Exception as e:
            logger.error(f"FastDetector: Error in analyze_user_comments: {str(e)}")
            logger.error(f"FastDetector: Traceback: {traceback.format_exc()}")
            raise e
    
    def should_skip_deep_analysis(self, fast_results: Dict) -> bool:
        """
        Determine if deep analysis should be skipped based on fast results.
        
        Args:
            fast_results: Results from fast screening
            
        Returns:
            True if deep analysis should be skipped
        """
        bot_score = fast_results.get("bot_score", 0.0) / 100.0  # Convert back to 0-1
        
        return (
            bot_score < settings.fast_threshold_low or 
            bot_score > settings.fast_threshold_high
        )
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "pipeline" if self.pipeline else "manual",
            "max_sequence_length": settings.max_sequence_length
        }
