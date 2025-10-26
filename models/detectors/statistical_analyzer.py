"""
Statistical analysis components for bot detection including perplexity, BPC, and linguistic features.
"""
import time
import warnings
import numpy as np
from typing import List, Dict, Optional, Union

# Suppress sentencepiece tokenizer conversion warning
warnings.filterwarnings("ignore", message=".*sentencepiece tokenizer.*byte fallback.*")

from config.settings import settings
from utils.preprocessing import RedditPreprocessor
from core.device_manager import DeviceManager
from utils.metrics import (
    PerplexityCalculator, 
    BPCCalculator, 
    SentimentAnalyzer, 
    EmbeddingSimilarityAnalyzer,
    ReadabilityAnalyzer,
    ZeroShotClassifier
)


class StatisticalAnalyzer:
    """Statistical analysis for bot detection using multiple signals."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            device: Device to run models on
        """
        self.device_manager = DeviceManager(device)
        self.device = str(self.device_manager.get_device())
        self.preprocessor = RedditPreprocessor(max_length=settings.max_sequence_length)
        
        # Initialize analyzers
        self.perplexity_calc = None
        self.bpc_calc = None
        self.sentiment_analyzer = None
        self.embedding_analyzer = None
        self.readability_analyzer = None
        self.zero_shot_classifier = None
        
        # Load models
        self._load_analyzers()
    
    def _load_analyzers(self):
        """Load all statistical analyzers."""
        try:
            print("Loading statistical analyzers...")
            
            # Perplexity calculator
            self.perplexity_calc = PerplexityCalculator(
                model_name=settings.perplexity_model_name,
                device=self.device
            )
            print("Perplexity calculator loaded")
            
            # BPC calculator
            self.bpc_calc = BPCCalculator()
            print("BPC calculator loaded")
            
            # Sentiment analyzer
            self.sentiment_analyzer = SentimentAnalyzer()
            print("Sentiment analyzer loaded")
            
            # Embedding analyzer
            self.embedding_analyzer = EmbeddingSimilarityAnalyzer(
                model_name=settings.embedding_model_name
            )
            print("Embedding analyzer loaded")
            
            # Readability analyzer
            self.readability_analyzer = ReadabilityAnalyzer()
            print("Readability analyzer loaded")
            
            # Zero-shot classifier
            self.zero_shot_classifier = ZeroShotClassifier()
            print("Zero-shot classifier loaded")
            
        except Exception as e:
            print(f"Some analyzers failed to load: {e}")
            # Continue with available analyzers
    
    def analyze_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze perplexity scores for texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Perplexity analysis results
        """
        if not self.perplexity_calc or not texts:
            return {"mean_perplexity": 0.0, "perplexity_std": 0.0, "bot_signal": 0.0}
        
        try:
            # Clean texts first
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            
            # Calculate perplexities
            perplexities = self.perplexity_calc.calculate_batch_perplexity(cleaned_texts)
            
            # Filter out infinite values
            valid_perplexities = [p for p in perplexities if p != float('inf')]
            
            if not valid_perplexities:
                return {"mean_perplexity": 0.0, "perplexity_std": 0.0, "bot_signal": 0.0}
            
            mean_perplexity = np.mean(valid_perplexities)
            perplexity_std = np.std(valid_perplexities)
            
            # AI text typically has lower perplexity (more predictable)
            # Convert to bot signal (0-1, higher = more bot-like)
            # Typical human text has perplexity 50-200, AI text 20-80
            if mean_perplexity < 50:
                bot_signal = 0.8  # Very low perplexity = likely AI
            elif mean_perplexity < 100:
                bot_signal = 0.6  # Low perplexity = possibly AI
            elif mean_perplexity < 150:
                bot_signal = 0.3  # Medium perplexity = possibly human
            else:
                bot_signal = 0.1  # High perplexity = likely human
            
            return {
                "mean_perplexity": mean_perplexity,
                "perplexity_std": perplexity_std,
                "bot_signal": bot_signal,
                "individual_perplexities": valid_perplexities
            }
            
        except Exception as e:
            print(f"Error in perplexity analysis: {e}")
            return {"mean_perplexity": 0.0, "perplexity_std": 0.0, "bot_signal": 0.0}
    
    def analyze_bpc(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze Bits Per Character for texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            BPC analysis results
        """
        if not self.bpc_calc or not texts:
            return {"mean_bpc": 0.0, "bpc_std": 0.0, "bot_signal": 0.0}
        
        try:
            # Clean texts first
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            
            # Calculate BPC scores
            bpc_scores = self.bpc_calc.calculate_batch_bpc(cleaned_texts)
            
            if not bpc_scores:
                return {"mean_bpc": 0.0, "bpc_std": 0.0, "bot_signal": 0.0}
            
            mean_bpc = np.mean(bpc_scores)
            bpc_std = np.std(bpc_scores)
            
            # AI text often has different information density
            # This is a heuristic - may need tuning based on data
            if mean_bpc < 3.0:
                bot_signal = 0.7  # Very low BPC = possibly AI
            elif mean_bpc < 4.0:
                bot_signal = 0.5  # Low BPC = possibly AI
            elif mean_bpc < 5.0:
                bot_signal = 0.3  # Medium BPC = possibly human
            else:
                bot_signal = 0.1  # High BPC = likely human
            
            return {
                "mean_bpc": mean_bpc,
                "bpc_std": bpc_std,
                "bot_signal": bot_signal,
                "individual_bpc": bpc_scores
            }
            
        except Exception as e:
            print(f"Error in BPC analysis: {e}")
            return {"mean_bpc": 0.0, "bpc_std": 0.0, "bot_signal": 0.0}
    
    def analyze_sentiment_consistency(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment consistency across texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Sentiment consistency analysis results
        """
        if not self.sentiment_analyzer or not texts:
            return {"consistency_score": 0.0, "variance": 0.0, "bot_signal": 0.0}
        
        try:
            # Clean texts first
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            
            # Calculate sentiment consistency
            sentiment_metrics = self.sentiment_analyzer.calculate_sentiment_consistency(cleaned_texts)
            
            # Higher consistency = more bot-like
            bot_signal = sentiment_metrics.get("consistency_score", 0.0)
            
            return {
                "consistency_score": sentiment_metrics.get("consistency_score", 0.0),
                "variance": sentiment_metrics.get("variance", 0.0),
                "mean_sentiment": sentiment_metrics.get("mean_sentiment", 0.0),
                "sentiment_range": sentiment_metrics.get("sentiment_range", 0.0),
                "bot_signal": bot_signal
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {"consistency_score": 0.0, "variance": 0.0, "bot_signal": 0.0}
    
    def analyze_embedding_similarity(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze embedding similarity between texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Embedding similarity analysis results
        """
        if not self.embedding_analyzer or not texts or len(texts) < 2:
            return {"mean_similarity": 0.0, "similarity_variance": 0.0, "bot_signal": 0.0}
        
        try:
            # Clean texts first
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            
            # Calculate similarity metrics
            similarity_metrics = self.embedding_analyzer.calculate_similarity_metrics(cleaned_texts)
            
            # Higher similarity = more bot-like (repetitive content)
            bot_signal = similarity_metrics.get("bot_likelihood", 0.0)
            
            return {
                "mean_similarity": similarity_metrics.get("mean_similarity", 0.0),
                "similarity_variance": similarity_metrics.get("similarity_variance", 0.0),
                "max_similarity": similarity_metrics.get("max_similarity", 0.0),
                "min_similarity": similarity_metrics.get("min_similarity", 0.0),
                "bot_signal": bot_signal
            }
            
        except Exception as e:
            print(f"Error in embedding similarity analysis: {e}")
            return {"mean_similarity": 0.0, "similarity_variance": 0.0, "bot_signal": 0.0}
    
    def analyze_readability(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze readability metrics for texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Readability analysis results
        """
        if not self.readability_analyzer or not texts:
            return {"mean_flesch_kincaid": 0.0, "readability_std": 0.0, "bot_signal": 0.0}
        
        try:
            # Clean texts first
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            
            # Calculate readability metrics
            readability_metrics = self.readability_analyzer.calculate_batch_readability(cleaned_texts)
            
            # Extract Flesch-Kincaid scores
            fk_scores = [metrics.get("flesch_kincaid", 0.0) for metrics in readability_metrics]
            
            if not fk_scores:
                return {"mean_flesch_kincaid": 0.0, "readability_std": 0.0, "bot_signal": 0.0}
            
            mean_fk = np.mean(fk_scores)
            fk_std = np.std(fk_scores)
            
            # AI text often has specific readability patterns
            # This is a heuristic - may need tuning
            if mean_fk < 5.0:
                bot_signal = 0.6  # Very low readability = possibly AI
            elif mean_fk < 10.0:
                bot_signal = 0.4  # Low readability = possibly AI
            elif mean_fk < 15.0:
                bot_signal = 0.2  # Medium readability = possibly human
            else:
                bot_signal = 0.1  # High readability = likely human
            
            return {
                "mean_flesch_kincaid": mean_fk,
                "readability_std": fk_std,
                "bot_signal": bot_signal,
                "individual_fk_scores": fk_scores
            }
            
        except Exception as e:
            print(f"Error in readability analysis: {e}")
            return {"mean_flesch_kincaid": 0.0, "readability_std": 0.0, "bot_signal": 0.0}
    
    def analyze_zero_shot(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze texts using zero-shot classification.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Zero-shot analysis results
        """
        if not self.zero_shot_classifier or not texts:
            return {"mean_ai_score": 0.0, "ai_score_std": 0.0, "bot_signal": 0.0}
        
        try:
            # Clean texts first
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=False)
            cleaned_texts = preprocessed['cleaned_comments']
            
            # Classify texts
            classifications = self.zero_shot_classifier.classify_batch(cleaned_texts)
            
            # Extract AI scores
            ai_scores = []
            for classification in classifications:
                ai_score = classification.get("written by AI", 0.0)
                if ai_score == 0.0:
                    # Try alternative labels
                    ai_score = classification.get("written by bot", 0.0)
                ai_scores.append(ai_score)
            
            if not ai_scores:
                return {"mean_ai_score": 0.0, "ai_score_std": 0.0, "bot_signal": 0.0}
            
            mean_ai_score = np.mean(ai_scores)
            ai_score_std = np.std(ai_scores)
            
            # Direct bot signal from AI classification
            bot_signal = mean_ai_score
            
            return {
                "mean_ai_score": mean_ai_score,
                "ai_score_std": ai_score_std,
                "bot_signal": bot_signal,
                "individual_ai_scores": ai_scores
            }
            
        except Exception as e:
            print(f"Error in zero-shot analysis: {e}")
            return {"mean_ai_score": 0.0, "ai_score_std": 0.0, "bot_signal": 0.0}
    
    def analyze_linguistic_features(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze linguistic features from preprocessing.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Linguistic feature analysis results
        """
        if not texts:
            return {"burstiness": 0.0, "lexical_diversity": 0.0, "emoji_ratio": 0.0, "em_dash_ratio": 0.0, "bot_signal": 0.0}
        
        try:
            # Get linguistic features
            preprocessed = self.preprocessor.batch_preprocess(texts, include_features=True)
            features = preprocessed['features']
            
            if not features:
                return {"burstiness": 0.0, "lexical_diversity": 0.0, "emoji_ratio": 0.0, "em_dash_ratio": 0.0, "bot_signal": 0.0}
            
            # Aggregate features
            burstiness_scores = [f.get('burstiness', 0.0) for f in features]
            lexical_diversity_scores = [f.get('lexical_diversity', 0.0) for f in features]
            emoji_counts = [f.get('emoji_count', 0) for f in features]
            em_dash_counts = [f.get('em_dash_count', 0) for f in features]
            word_counts = [f.get('word_count', 1) for f in features]
            
            # Calculate ratios
            emoji_ratios = [emoji / max(word, 1) for emoji, word in zip(emoji_counts, word_counts)]
            em_dash_ratios = [em_dash / max(word, 1) for em_dash, word in zip(em_dash_counts, word_counts)]
            
            # Calculate means
            mean_burstiness = np.mean(burstiness_scores)
            mean_lexical_diversity = np.mean(lexical_diversity_scores)
            mean_emoji_ratio = np.mean(emoji_ratios)
            mean_em_dash_ratio = np.mean(em_dash_ratios)
            
            # Calculate bot signals
            # Lower burstiness = more bot-like (uniform sentence lengths)
            burstiness_signal = max(0.0, 1.0 - mean_burstiness)
            
            # Lower lexical diversity = more bot-like (repetitive vocabulary)
            diversity_signal = max(0.0, 1.0 - mean_lexical_diversity)
            
            # Higher emoji ratio = more bot-like (bots overuse emojis)
            emoji_signal = min(1.0, mean_emoji_ratio * 10)  # Scale up emoji ratio
            
            # Higher em dash ratio = more bot-like (AI models love em dashes)
            em_dash_signal = min(1.0, mean_em_dash_ratio * 20)  # Scale up em dash ratio
            
            # Combined linguistic bot signal
            linguistic_bot_signal = (burstiness_signal + diversity_signal + emoji_signal + em_dash_signal) / 4.0
            
            return {
                "burstiness": mean_burstiness,
                "lexical_diversity": mean_lexical_diversity,
                "emoji_ratio": mean_emoji_ratio,
                "em_dash_ratio": mean_em_dash_ratio,
                "bot_signal": linguistic_bot_signal,
                "burstiness_signal": burstiness_signal,
                "diversity_signal": diversity_signal,
                "emoji_signal": emoji_signal,
                "em_dash_signal": em_dash_signal
            }
            
        except Exception as e:
            print(f"Error in linguistic feature analysis: {e}")
            return {"burstiness": 0.0, "lexical_diversity": 0.0, "emoji_ratio": 0.0, "em_dash_ratio": 0.0, "bot_signal": 0.0}
    
    def analyze_all_signals(self, texts: List[str]) -> Dict[str, Union[float, Dict]]:
        """
        Run all statistical analyses on texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Complete statistical analysis results
        """
        start_time = time.time()
        
        if not texts:
            return {
                "perplexity": {"bot_signal": 0.0},
                "bpc": {"bot_signal": 0.0},
                "sentiment_consistency": {"bot_signal": 0.0},
                "embedding_similarity": {"bot_signal": 0.0},
                "readability": {"bot_signal": 0.0},
                "zero_shot": {"bot_signal": 0.0},
                "linguistic_features": {"bot_signal": 0.0},
                "processing_time_ms": 0.0
            }
        
        # Run all analyses
        results = {
            "perplexity": self.analyze_perplexity(texts),
            "bpc": self.analyze_bpc(texts),
            "sentiment_consistency": self.analyze_sentiment_consistency(texts),
            "embedding_similarity": self.analyze_embedding_similarity(texts),
            "readability": self.analyze_readability(texts),
            "zero_shot": self.analyze_zero_shot(texts),
            "linguistic_features": self.analyze_linguistic_features(texts)
        }
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        results["processing_time_ms"] = processing_time
        
        return results
    
    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzers."""
        analyzers = []
        if self.perplexity_calc:
            analyzers.append("perplexity")
        if self.bpc_calc:
            analyzers.append("bpc")
        if self.sentiment_analyzer:
            analyzers.append("sentiment_consistency")
        if self.embedding_analyzer:
            analyzers.append("embedding_similarity")
        if self.readability_analyzer:
            analyzers.append("readability")
        if self.zero_shot_classifier:
            analyzers.append("zero_shot")
        analyzers.append("linguistic_features")  # Always available
        return analyzers
