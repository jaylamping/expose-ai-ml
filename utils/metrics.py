"""
Statistical metrics for bot detection including perplexity and BPC calculations.
"""
import math
import numpy as np
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import textstat
from scipy import stats


class PerplexityCalculator:
    """Calculate perplexity scores for text using language models."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        """
        Initialize perplexity calculator.
        
        Args:
            model_name: Name of the language model to use
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity for a single text.
        
        Args:
            text: Text to calculate perplexity for
            
        Returns:
            Perplexity score
        """
        if not text.strip():
            return float('inf')
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_batch_perplexity(self, texts: List[str]) -> List[float]:
        """
        Calculate perplexity for a batch of texts.
        
        Args:
            texts: List of texts to calculate perplexity for
            
        Returns:
            List of perplexity scores
        """
        perplexities = []
        
        for text in texts:
            try:
                perplexity = self.calculate_perplexity(text)
                perplexities.append(perplexity)
            except Exception as e:
                print(f"Error calculating perplexity for text: {e}")
                perplexities.append(float('inf'))
        
        return perplexities


class BPCCalculator:
    """Calculate Bits Per Character (BPC) for text compression analysis."""
    
    def __init__(self):
        """Initialize BPC calculator."""
        pass
    
    def calculate_bpc(self, text: str) -> float:
        """
        Calculate Bits Per Character for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            BPC score
        """
        if not text:
            return 0.0
        
        # Simple character-level entropy calculation
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_batch_bpc(self, texts: List[str]) -> List[float]:
        """
        Calculate BPC for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of BPC scores
        """
        return [self.calculate_bpc(text) for text in texts]


class SentimentAnalyzer:
    """Analyze sentiment consistency across comments."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: Sentiment analysis model to use
        """
        from transformers import pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment scores
        """
        if not text.strip():
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        
        try:
            results = self.sentiment_pipeline(text)
            sentiment_scores = {}
            for result in results[0]:
                sentiment_scores[result['label'].lower()] = result['score']
            return sentiment_scores
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    
    def calculate_sentiment_consistency(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate sentiment consistency across multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Sentiment consistency metrics
        """
        if not texts:
            return {"variance": 0.0, "consistency_score": 0.0}
        
        sentiments = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            # Convert to single score (positive - negative)
            score = sentiment.get("positive", 0.0) - sentiment.get("negative", 0.0)
            sentiments.append(score)
        
        if len(sentiments) < 2:
            return {"variance": 0.0, "consistency_score": 1.0}
        
        variance = np.var(sentiments)
        # Lower variance = higher consistency (more bot-like)
        consistency_score = 1.0 / (1.0 + variance)
        
        return {
            "variance": variance,
            "consistency_score": consistency_score,
            "mean_sentiment": np.mean(sentiments),
            "sentiment_range": np.max(sentiments) - np.min(sentiments)
        }


class EmbeddingSimilarityAnalyzer:
    """Analyze embedding similarity between comments."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding analyzer.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def calculate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Calculate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts)
        return embeddings
    
    def calculate_similarity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate similarity metrics between comments.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Similarity metrics
        """
        if len(texts) < 2:
            return {"mean_similarity": 0.0, "similarity_variance": 0.0}
        
        embeddings = self.calculate_embeddings(texts)
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(similarity)
        
        if not similarities:
            return {"mean_similarity": 0.0, "similarity_variance": 0.0}
        
        mean_similarity = np.mean(similarities)
        similarity_variance = np.var(similarities)
        
        # High similarity = more bot-like (repetitive content)
        bot_likelihood = mean_similarity
        
        return {
            "mean_similarity": mean_similarity,
            "similarity_variance": similarity_variance,
            "bot_likelihood": bot_likelihood,
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities)
        }


class ReadabilityAnalyzer:
    """Analyze readability metrics for text."""
    
    def __init__(self):
        """Initialize readability analyzer."""
        pass
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability metrics
        """
        if not text.strip():
            return {
                "flesch_kincaid": 0.0,
                "gunning_fog": 0.0,
                "smog": 0.0,
                "automated_readability": 0.0
            }
        
        try:
            return {
                "flesch_kincaid": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog": textstat.smog_index(text),
                "automated_readability": textstat.automated_readability_index(text)
            }
        except Exception as e:
            print(f"Error calculating readability: {e}")
            return {
                "flesch_kincaid": 0.0,
                "gunning_fog": 0.0,
                "smog": 0.0,
                "automated_readability": 0.0
            }
    
    def calculate_batch_readability(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Calculate readability metrics for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of readability metrics
        """
        return [self.calculate_readability_metrics(text) for text in texts]


class ZeroShotClassifier:
    """Zero-shot classification for bot detection."""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name: Zero-shot classification model to use
        """
        from transformers import pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name
        )
    
    def classify_text(self, text: str, labels: List[str] = None) -> Dict[str, float]:
        """
        Classify text using zero-shot classification.
        
        Args:
            text: Text to classify
            labels: Classification labels
            
        Returns:
            Classification scores
        """
        if labels is None:
            labels = ["written by human", "written by AI", "written by bot"]
        
        if not text.strip():
            return {label: 0.0 for label in labels}
        
        try:
            result = self.classifier(text, labels)
            scores = {}
            for i, label in enumerate(result['labels']):
                scores[label] = result['scores'][i]
            return scores
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return {label: 0.0 for label in labels}
    
    def classify_batch(self, texts: List[str], labels: List[str] = None) -> List[Dict[str, float]]:
        """
        Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            labels: Classification labels
            
        Returns:
            List of classification scores
        """
        return [self.classify_text(text, labels) for text in texts]


def calculate_ensemble_confidence(scores: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate ensemble confidence based on score agreement.
    
    Args:
        scores: Dictionary of individual scores
        weights: Dictionary of weights for each score
        
    Returns:
        Confidence metrics
    """
    # Calculate weighted average
    weighted_sum = sum(scores.get(key, 0.0) * weights.get(key, 0.0) for key in weights.keys())
    weight_sum = sum(weights.values())
    weighted_average = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    # Calculate confidence based on agreement
    score_values = [scores.get(key, 0.0) for key in weights.keys() if key in scores]
    if len(score_values) > 1:
        agreement = 1.0 - np.std(score_values)  # Lower std = higher agreement
        agreement = max(0.0, min(1.0, agreement))  # Clamp to [0, 1]
    else:
        agreement = 1.0
    
    return {
        "weighted_score": weighted_average,
        "confidence": agreement,
        "score_variance": np.var(score_values) if len(score_values) > 1 else 0.0,
        "individual_scores": scores
    }
