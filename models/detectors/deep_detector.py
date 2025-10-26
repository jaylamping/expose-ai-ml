"""
Deep analysis model for bot detection using DeBERTa-v3-base.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time
from pathlib import Path

from config.settings import settings
from utils.preprocessing import RedditPreprocessor
from core.device_manager import DeviceManager


class DeepDetector:
    """Deep analysis model for high-accuracy bot detection."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the deep detector.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
        """
        self.model_name = model_name or settings.deep_model_name
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
            # Try to load as pipeline first (simpler)
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if "cuda" in self.device else -1,
                return_all_scores=True
            )
            print(f"Loaded deep detector pipeline: {self.model_name}")
        except Exception as e:
            print(f"Pipeline loading failed, trying manual loading: {e}")
            try:
                # Manual loading as fallback
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model = self.device_manager.to_device(self.model)
                self.model.eval()
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"Loaded deep detector manually: {self.model_name}")
            except Exception as e2:
                print(f"Failed to load deep detector: {e2}")
                # For now, we'll use a fallback model
                self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails."""
        try:
            # Use a more commonly available model as fallback
            fallback_model = "distilbert-base-uncased"
            print(f"Loading fallback model: {fallback_model}")
            
            self.pipeline = pipeline(
                "text-classification",
                model=fallback_model,
                device=0 if "cuda" in self.device else -1,
                return_all_scores=True
            )
            self.model_name = fallback_model
            print(f"Loaded fallback deep detector: {fallback_model}")
        except Exception as e:
            print(f"Failed to load fallback model: {e}")
            raise e
    
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
            print(f"Error in deep detection: {e}")
            return {"bot_probability": 0.0, "confidence": 0.0, "error": str(e)}
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict bot probability for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction results
        """
        if not texts:
            return []
        
        # Clean texts
        preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
        cleaned_texts = preprocessed['cleaned_comments']
        
        results = []
        
        try:
            if self.pipeline:
                # Batch processing with pipeline
                batch_results = self.pipeline(cleaned_texts)
                
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
                # Manual batch processing
                inputs = self.tokenizer(
                    cleaned_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=settings.max_sequence_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    
                    for i in range(len(texts)):
                        bot_prob = probabilities[i][1].item()  # Assuming class 1 is bot
                        confidence = torch.max(probabilities[i]).item()
                        
                        results.append({
                            "bot_probability": bot_prob,
                            "confidence": confidence,
                            "model": self.model_name
                        })
        
        except Exception as e:
            print(f"Error in batch deep detection: {e}")
            # Return default results for all texts
            results = [{"bot_probability": 0.0, "confidence": 0.0, "error": str(e)} for _ in texts]
        
        return results
    
    def analyze_user_comments(self, comments: List[str], use_sampling: bool = True) -> Dict[str, Union[float, List, Dict]]:
        """
        Analyze a user's comments with deep analysis.
        
        Args:
            comments: List of user comments
            use_sampling: Whether to use smart sampling for large comment sets
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        
        # Limit comments if too many
        if len(comments) > settings.max_comments_per_request:
            comments = comments[:settings.max_comments_per_request]
        
        # Use smart sampling if requested and we have many comments
        if use_sampling and len(comments) > 30:
            sampled_comments = self.preprocessor.smart_sampling(comments, sample_size=30)
        else:
            sampled_comments = comments
        
        # Get predictions
        predictions = self.predict_batch(sampled_comments)
        
        # Extract bot probabilities
        bot_probs = [pred["bot_probability"] for pred in predictions]
        confidences = [pred["confidence"] for pred in predictions]
        
        # Calculate aggregate scores
        mean_bot_prob = np.mean(bot_probs) if bot_probs else 0.0
        mean_confidence = np.mean(confidences) if confidences else 0.0
        std_bot_prob = np.std(bot_probs) if len(bot_probs) > 1 else 0.0
        
        # Calculate additional metrics
        high_confidence_predictions = [p for p in bot_probs if p > 0.8 or p < 0.2]
        consistency_score = len(high_confidence_predictions) / len(bot_probs) if bot_probs else 0.0
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "bot_score": mean_bot_prob * 100,  # Convert to percentage
            "confidence": mean_confidence * 100,
            "processing_time_ms": processing_time,
            "comments_analyzed": len(sampled_comments),
            "total_comments": len(comments),
            "individual_scores": bot_probs,
            "score_std": std_bot_prob,
            "consistency_score": consistency_score,
            "model": self.model_name,
            "stage": "deep_analysis"
        }
    
    def analyze_with_context(self, comments: List[str], parent_contexts: List[Dict]) -> Dict[str, Union[float, List, Dict]]:
        """
        Analyze comments with parent context for better accuracy.
        
        Args:
            comments: List of user comments
            parent_contexts: List of parent context dictionaries
            
        Returns:
            Analysis results with context
        """
        start_time = time.time()
        
        # Create comment-context pairs
        formatted_pairs = self.preprocessor.create_context_pairs(comments, parent_contexts)
        
        # Limit if too many
        if len(formatted_pairs) > settings.max_comments_per_request:
            formatted_pairs = formatted_pairs[:settings.max_comments_per_request]
        
        # Use smart sampling for large sets
        if len(formatted_pairs) > 20:
            sampled_pairs = self.preprocessor.smart_sampling(formatted_pairs, sample_size=20)
        else:
            sampled_pairs = formatted_pairs
        
        # Get predictions
        predictions = self.predict_batch(sampled_pairs)
        
        # Extract bot probabilities
        bot_probs = [pred["bot_probability"] for pred in predictions]
        confidences = [pred["confidence"] for pred in predictions]
        
        # Calculate aggregate scores
        mean_bot_prob = np.mean(bot_probs) if bot_probs else 0.0
        mean_confidence = np.mean(confidences) if confidences else 0.0
        std_bot_prob = np.std(bot_probs) if len(bot_probs) > 1 else 0.0
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "bot_score": mean_bot_prob * 100,  # Convert to percentage
            "confidence": mean_confidence * 100,
            "processing_time_ms": processing_time,
            "comments_analyzed": len(sampled_pairs),
            "total_comments": len(comments),
            "individual_scores": bot_probs,
            "score_std": std_bot_prob,
            "model": self.model_name,
            "stage": "deep_analysis_with_context",
            "context_used": True
        }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "pipeline" if self.pipeline else "manual",
            "max_sequence_length": settings.max_sequence_length
        }
