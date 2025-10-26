"""
Ensemble scoring system for combining multiple bot detection signals.
"""
import time
import numpy as np
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass

from config.settings import settings
from utils.metrics import calculate_ensemble_confidence


@dataclass
class AnalysisResult:
    """Container for analysis results from different stages."""
    stage: str
    bot_score: float  # 0-100
    confidence: float  # 0-100
    processing_time_ms: float
    breakdown: Optional[Dict[str, Any]] = None
    should_skip_next: bool = False


class EnsembleScorer:
    """Ensemble scoring system for bot detection."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble scorer.
        
        Args:
            weights: Custom weights for ensemble scoring
        """
        self.weights = weights or settings.ensemble_weights.copy()
        self.threshold = settings.bot_threshold
        
        # Validate weights
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0."""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Ensemble weights sum to {total_weight:.3f}, not 1.0")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total_weight
            print(f"Normalized weights: {self.weights}")
    
    def combine_fast_and_deep(self, fast_result: AnalysisResult, deep_result: Optional[AnalysisResult] = None) -> Dict[str, Union[float, Dict]]:
        """
        Combine fast and deep model results.
        
        Args:
            fast_result: Results from fast screening
            deep_result: Results from deep analysis (optional)
            
        Returns:
            Combined results
        """
        start_time = time.time()
        
        # If no deep result, use fast result only
        if deep_result is None:
            return {
                "bot_score": fast_result.bot_score,
                "confidence": fast_result.confidence,
                "stage": "fast_only",
                "processing_time_ms": fast_result.processing_time_ms,
                "breakdown": {
                    "fast_model": fast_result.bot_score,
                    "deep_model": None
                },
                "is_likely_bot": fast_result.bot_score > (self.threshold * 100)
            }
        
        # Combine fast and deep results
        fast_weight = self.weights.get("fast_model", 0.25)
        deep_weight = self.weights.get("deep_model", 0.30)
        
        # Normalize weights for just these two models
        total_weight = fast_weight + deep_weight
        fast_weight_norm = fast_weight / total_weight
        deep_weight_norm = deep_weight / total_weight
        
        # Calculate weighted average
        combined_score = (fast_result.bot_score * fast_weight_norm + 
                         deep_result.bot_score * deep_weight_norm)
        
        # Calculate combined confidence
        combined_confidence = (fast_result.confidence * fast_weight_norm + 
                              deep_result.confidence * deep_weight_norm)
        
        processing_time = (time.time() - start_time) * 1000 + fast_result.processing_time_ms + deep_result.processing_time_ms
        
        return {
            "bot_score": combined_score,
            "confidence": combined_confidence,
            "stage": "fast_and_deep",
            "processing_time_ms": processing_time,
            "breakdown": {
                "fast_model": fast_result.bot_score,
                "deep_model": deep_result.bot_score
            },
            "is_likely_bot": combined_score > (self.threshold * 100)
        }
    
    def calculate_ensemble_score(self, scores: Dict[str, float]) -> Dict[str, Union[float, Dict]]:
        """
        Calculate ensemble score from individual model scores.
        
        Args:
            scores: Dictionary of model scores
            
        Returns:
            Ensemble score and confidence
        """
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for model_name, score in scores.items():
            weight = self.weights.get(model_name, 0.0)
            total_score += score * weight
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            ensemble_score = total_score / total_weight
        else:
            ensemble_score = sum(scores.values()) / len(scores)
        
        # Calculate confidence based on score consistency
        if len(scores) > 1:
            score_values = list(scores.values())
            variance = sum((x - ensemble_score) ** 2 for x in score_values) / len(score_values)
            confidence = max(0, 100 - (variance * 100))
        else:
            confidence = 80.0  # Default confidence for single model
        
        return {
            "ensemble_score": ensemble_score,
            "confidence": confidence,
            "breakdown": scores,
            "is_likely_bot": ensemble_score > (self.threshold * 100)
        }
    
    def combine_all_signals(self, 
                           fast_result: AnalysisResult,
                           deep_result: Optional[AnalysisResult] = None,
                           statistical_results: Optional[Dict] = None) -> Dict[str, Union[float, Dict]]:
        """
        Combine all available signals for final scoring.
        
        Args:
            fast_result: Results from fast screening
            deep_result: Results from deep analysis (optional)
            statistical_results: Results from statistical analysis (optional)
            
        Returns:
            Complete ensemble results
        """
        start_time = time.time()
        
        # Start with fast and deep combination
        base_result = self.combine_fast_and_deep(fast_result, deep_result)
        
        # If no statistical results, return base result
        if not statistical_results:
            return base_result
        
        # Extract individual signals
        signals = {
            "fast_model": fast_result.bot_score / 100.0,  # Convert to 0-1
            "deep_model": (deep_result.bot_score / 100.0) if deep_result else 0.0,
            "perplexity": statistical_results.get("perplexity", {}).get("bot_signal", 0.0),
            "bpc": statistical_results.get("bpc", {}).get("bot_signal", 0.0),
            "sentiment_consistency": statistical_results.get("sentiment_consistency", {}).get("bot_signal", 0.0),
            "embedding_similarity": statistical_results.get("embedding_similarity", {}).get("bot_signal", 0.0),
            "zero_shot": statistical_results.get("zero_shot", {}).get("bot_signal", 0.0),
            "burstiness": statistical_results.get("linguistic_features", {}).get("bot_signal", 0.0)
        }
        
        # Calculate ensemble confidence
        confidence_result = calculate_ensemble_confidence(signals, self.weights)
        
        # Calculate final weighted score
        final_score = confidence_result["weighted_score"] * 100  # Convert back to 0-100
        
        # Calculate processing time
        statistical_time = statistical_results.get("processing_time_ms", 0.0)
        processing_time = (time.time() - start_time) * 1000 + base_result["processing_time_ms"] + statistical_time
        
        # Create detailed breakdown
        breakdown = {
            "fast_model": fast_result.bot_score,
            "deep_model": deep_result.bot_score if deep_result else None,
            "perplexity": signals["perplexity"] * 100,
            "bpc": signals["bpc"] * 100,
            "sentiment_consistency": signals["sentiment_consistency"] * 100,
            "embedding_similarity": signals["embedding_similarity"] * 100,
            "zero_shot": signals["zero_shot"] * 100,
            "burstiness": signals["burstiness"] * 100,
            "weights": self.weights,
            "individual_scores": confidence_result["individual_scores"]
        }
        
        return {
            "bot_score": final_score,
            "confidence": confidence_result["confidence"] * 100,
            "stage": "full_ensemble",
            "processing_time_ms": processing_time,
            "breakdown": breakdown,
            "is_likely_bot": final_score > (self.threshold * 100),
            "score_variance": confidence_result["score_variance"],
            "agreement_score": confidence_result["confidence"]
        }
    
    def calculate_adaptive_weights(self, 
                                 fast_result: AnalysisResult,
                                 deep_result: Optional[AnalysisResult] = None,
                                 statistical_results: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate adaptive weights based on confidence and agreement.
        
        Args:
            fast_result: Results from fast screening
            deep_result: Results from deep analysis (optional)
            statistical_results: Results from statistical analysis (optional)
            
        Returns:
            Adaptive weights
        """
        adaptive_weights = self.weights.copy()
        
        # If fast model has very high confidence, increase its weight
        if fast_result.confidence > 90:
            adaptive_weights["fast_model"] *= 1.2
            if deep_result:
                adaptive_weights["deep_model"] *= 0.8
        
        # If deep model has very high confidence, increase its weight
        if deep_result and deep_result.confidence > 90:
            adaptive_weights["deep_model"] *= 1.2
            adaptive_weights["fast_model"] *= 0.8
        
        # If statistical signals agree strongly, increase their weights
        if statistical_results:
            statistical_signals = [
                statistical_results.get("perplexity", {}).get("bot_signal", 0.0),
                statistical_results.get("bpc", {}).get("bot_signal", 0.0),
                statistical_results.get("sentiment_consistency", {}).get("bot_signal", 0.0),
                statistical_results.get("embedding_similarity", {}).get("bot_signal", 0.0)
            ]
            
            # Calculate agreement among statistical signals
            if len(statistical_signals) > 1:
                statistical_std = np.std(statistical_signals)
                if statistical_std < 0.1:  # High agreement
                    for key in ["perplexity", "bpc", "sentiment_consistency", "embedding_similarity"]:
                        if key in adaptive_weights:
                            adaptive_weights[key] *= 1.1
        
        # Normalize weights
        total_weight = sum(adaptive_weights.values())
        for key in adaptive_weights:
            adaptive_weights[key] /= total_weight
        
        return adaptive_weights
    
    def get_confidence_explanation(self, result: Dict[str, Union[float, Dict]]) -> str:
        """
        Generate a human-readable explanation of the confidence score.
        
        Args:
            result: Ensemble result dictionary
            
        Returns:
            Explanation string
        """
        confidence = result.get("confidence", 0.0)
        bot_score = result.get("bot_score", 0.0)
        stage = result.get("stage", "unknown")
        
        if confidence > 90:
            conf_level = "very high"
        elif confidence > 75:
            conf_level = "high"
        elif confidence > 60:
            conf_level = "moderate"
        elif confidence > 40:
            conf_level = "low"
        else:
            conf_level = "very low"
        
        if bot_score > 70:
            classification = "likely bot"
        elif bot_score > 30:
            classification = "unclear"
        else:
            classification = "likely human"
        
        return f"Analysis using {stage} with {conf_level} confidence ({confidence:.1f}%). Result: {classification} ({bot_score:.1f}% bot score)."
    
    def should_skip_deep_analysis(self, fast_result: AnalysisResult) -> bool:
        """
        Determine if deep analysis should be skipped based on fast results.
        
        Args:
            fast_result: Results from fast screening
            
        Returns:
            True if deep analysis should be skipped
        """
        return fast_result.should_skip_next
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble configuration."""
        return {
            "weights": self.weights,
            "threshold": self.threshold,
            "total_weight": sum(self.weights.values()),
            "available_signals": list(self.weights.keys())
        }
