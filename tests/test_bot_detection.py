"""
Unit and integration tests for bot detection system.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path when not running via pytest
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import torch

from models.detectors.fast_detector import FastDetector
from models.detectors.deep_detector import DeepDetector
from models.detectors.statistical_analyzer import StatisticalAnalyzer
from models.ensemble import EnsembleScorer, AnalysisResult
from utils.preprocessing import RedditPreprocessor
from utils.metrics import PerplexityCalculator, BPCCalculator, SentimentAnalyzer
from api.bot_detection import BotDetectionAPI, AnalysisRequest, AnalysisOptions


class TestRedditPreprocessor:
    """Test Reddit comment preprocessing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = RedditPreprocessor()
    
    def test_clean_reddit_text(self):
        """Test Reddit text cleaning."""
        # Test URL removal
        text_with_url = "Check this out: https://reddit.com/r/test"
        cleaned = self.preprocessor.clean_reddit_text(text_with_url)
        assert "[URL]" in cleaned
        assert "https://reddit.com/r/test" not in cleaned
        
        # Test user mention normalization
        text_with_mention = "Hey u/username, what do you think?"
        cleaned = self.preprocessor.clean_reddit_text(text_with_mention)
        assert "[USER]" in cleaned
        assert "u/username" not in cleaned
        
        # Test subreddit normalization
        text_with_subreddit = "This is from r/AskReddit"
        cleaned = self.preprocessor.clean_reddit_text(text_with_subreddit)
        assert "[SUBREDDIT]" in cleaned
        assert "r/AskReddit" not in cleaned
        
        # Test markdown removal
        text_with_markdown = "**Bold text** and *italic text*"
        cleaned = self.preprocessor.clean_reddit_text(text_with_markdown)
        assert "Bold text" in cleaned
        assert "italic text" in cleaned
        assert "**" not in cleaned
        assert "*" not in cleaned
    
    def test_detect_bot_phrases(self):
        """Test bot phrase detection."""
        # Test bot phrases
        bot_text = "As an AI, I cannot have personal opinions on this matter."
        assert self.preprocessor.detect_bot_phrases(bot_text) == True
        
        # Test human text
        human_text = "I think this is a great idea!"
        assert self.preprocessor.detect_bot_phrases(human_text) == False
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction."""
        text = "This is a test sentence. It has multiple sentences!"
        features = self.preprocessor.extract_linguistic_features(text)
        
        assert features['length'] > 0
        assert features['word_count'] > 0
        assert features['sentence_count'] > 0
        assert features['avg_sentence_length'] > 0
        assert 'burstiness' in features
        assert 'lexical_diversity' in features
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        comments = [
            "This is a test comment.",
            "Another test comment with **markdown**."
        ]
        
        result = self.preprocessor.batch_preprocess(comments)
        
        assert len(result['cleaned_comments']) == 2
        assert len(result['features']) == 2
        assert len(result['bot_phrase_flags']) == 2
        assert "**" not in result['cleaned_comments'][1]


class TestPerplexityCalculator:
    """Test perplexity calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = PerplexityCalculator()
    
    def test_calculate_perplexity(self):
        """Test perplexity calculation."""
        text = "This is a test sentence."
        perplexity = self.calculator.calculate_perplexity(text)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert perplexity != float('inf')
    
    def test_calculate_batch_perplexity(self):
        """Test batch perplexity calculation."""
        texts = [
            "This is a test sentence.",
            "Another test sentence."
        ]
        
        perplexities = self.calculator.calculate_batch_perplexity(texts)
        
        assert len(perplexities) == 2
        assert all(isinstance(p, float) for p in perplexities)
        assert all(p > 0 for p in perplexities)


class TestBPCCalculator:
    """Test BPC calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = BPCCalculator()
    
    def test_calculate_bpc(self):
        """Test BPC calculation."""
        text = "This is a test sentence."
        bpc = self.calculator.calculate_bpc(text)
        
        assert isinstance(bpc, float)
        assert bpc >= 0
    
    def test_calculate_batch_bpc(self):
        """Test batch BPC calculation."""
        texts = [
            "This is a test sentence.",
            "Another test sentence."
        ]
        
        bpc_scores = self.calculator.calculate_batch_bpc(texts)
        
        assert len(bpc_scores) == 2
        assert all(isinstance(b, float) for b in bpc_scores)
        assert all(b >= 0 for b in bpc_scores)


class TestSentimentAnalyzer:
    """Test sentiment analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        text = "This is a great day!"
        sentiment = self.analyzer.analyze_sentiment(text)
        
        assert isinstance(sentiment, dict)
        assert 'positive' in sentiment
        assert 'negative' in sentiment
        assert 'neutral' in sentiment
    
    def test_calculate_sentiment_consistency(self):
        """Test sentiment consistency calculation."""
        texts = [
            "This is great!",
            "I love this!",
            "Amazing work!"
        ]
        
        consistency = self.analyzer.calculate_sentiment_consistency(texts)
        
        assert isinstance(consistency, dict)
        assert 'variance' in consistency
        assert 'consistency_score' in consistency


class TestEnsembleScorer:
    """Test ensemble scoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scorer = EnsembleScorer()
    
    def test_combine_fast_and_deep(self):
        """Test combining fast and deep model scores."""
        # Create AnalysisResult objects
        fast_result = AnalysisResult(
            stage="fast",
            bot_score=80.0,
            confidence=85.0,
            processing_time_ms=50.0
        )
        deep_result = AnalysisResult(
            stage="deep", 
            bot_score=70.0,
            confidence=90.0,
            processing_time_ms=200.0
        )
        
        result = self.scorer.combine_fast_and_deep(fast_result, deep_result)
        
        assert isinstance(result, dict)
        assert 'bot_score' in result
        assert 'confidence' in result
        assert 'stage' in result
        assert 'breakdown' in result
    
    def test_calculate_ensemble_score(self):
        """Test ensemble score calculation."""
        scores = {
            'fast_model': 0.8,
            'deep_model': 0.7,
            'perplexity': 0.6,
            'bpc': 0.5
        }
        
        result = self.scorer.calculate_ensemble_score(scores)
        
        assert isinstance(result, dict)
        assert 'ensemble_score' in result
        assert 'confidence' in result


class TestBotDetectionAPI:
    """Test bot detection API integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.api = BotDetectionAPI()
    
    @pytest.mark.asyncio
    async def test_full_analysis_integration(self):
        """Test full bot detection analysis integration."""
        # Create test comments with mix of human and bot-like content
        test_comments = [
            'This is a great post! I totally agree with your point of view.',
            'As an AI, I cannot have personal opinions on this matter.',
            'Thanks for sharing this information. Very helpful!',
            'I am designed to be helpful and provide accurate information.',
            'This is interesting. What do you think about the implications?'
        ]

        request = AnalysisRequest(
            user_id='test_user',
            comments=test_comments,
            options=AnalysisOptions()
        )

        # Initialize models
        await self.api._initialize_models()
        
        # Run analysis
        result = await self.api._analyze_user_comments(request)
        
        # Verify results
        assert isinstance(result, dict)
        assert 'user_id' in result
        assert 'bot_score' in result
        assert 'confidence' in result
        assert 'breakdown' in result
        
        assert result['user_id'] == 'test_user'
        assert 0 <= result['bot_score'] <= 100
        assert 0 <= result['confidence'] <= 100
        
        # Verify breakdown contains expected components
        breakdown = result['breakdown']
        expected_keys = ['fast_model', 'deep_model', 'perplexity', 'bpc']
        for key in expected_keys:
            assert key in breakdown
    
    @pytest.mark.asyncio
    async def test_fast_analysis_only(self):
        """Test fast analysis only mode."""
        test_comments = [
            'This is clearly a bot comment.',
            'I am an AI assistant.'
        ]

        request = AnalysisRequest(
            user_id='test_user_fast',
            comments=test_comments,
            options=AnalysisOptions(fast_only=True)
        )

        # Initialize models
        await self.api._initialize_models()
        
        # Run analysis
        result = await self.api._analyze_user_comments(request)
        
        # Verify results
        assert isinstance(result, dict)
        assert result['user_id'] == 'test_user_fast'
        assert 'bot_score' in result
        assert 'confidence' in result


def test_aggregate_linguistic_features():
    """Test linguistic feature aggregation."""
    from utils.preprocessing import aggregate_linguistic_features
    
    features = [
        {'length': 10, 'word_count': 5, 'sentence_count': 1},
        {'length': 20, 'word_count': 10, 'sentence_count': 2},
        {'length': 15, 'word_count': 8, 'sentence_count': 1}
    ]
    
    aggregated = aggregate_linguistic_features(features)
    
    assert isinstance(aggregated, dict)
    assert 'length_mean' in aggregated
    assert 'length_std' in aggregated
    assert 'word_count_mean' in aggregated


# Integration test that can be run standalone
async def run_integration_test():
    """Run a full integration test of the bot detection system."""
    print("Running Bot Detection Integration Test...")
    
    # Create test comments
    test_comments = [
        'This is a great post! I totally agree with your point of view.',
        'As an AI, I cannot have personal opinions on this matter.',
        'Thanks for sharing this information. Very helpful!',
        'I am designed to be helpful and provide accurate information.',
        'This is interesting. What do you think about the implications?'
    ]

    request = AnalysisRequest(
        user_id='test_user',
        comments=test_comments,
        options=AnalysisOptions()
    )

    # Initialize API
    api = BotDetectionAPI()
    
    # Initialize models first
    print('Initializing models...')
    await api._initialize_models()
    print('Models initialized successfully!')
    
    # Test the analysis
    try:
        result = await api._analyze_user_comments(request)
        print('\nBot Detection Test Results:')
        print(f'User ID: {result["user_id"]}')
        print(f'Bot Score: {result["bot_score"]:.3f}')
        print(f'Confidence: {result["confidence"]:.3f}')
        print(f'Analysis Time: {result.get("processing_time_ms", 0)/1000:.2f}s')
        print('\nBreakdown:')
        for key, value in result["breakdown"].items():
            if isinstance(value, float):
                print(f'  {key}: {value:.3f}')
            else:
                print(f'  {key}: {value}')
        print('\nIntegration test completed successfully!')
        return True
    except Exception as e:
        print(f'Integration test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run integration test when script is executed directly
    success = asyncio.run(run_integration_test())
    exit(0 if success else 1)