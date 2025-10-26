"""
Unit and integration tests for bot detection system.
"""
import pytest
import numpy as np
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
        bot_text = "As an AI, I don't have personal opinions"
        assert self.preprocessor.detect_bot_phrases(bot_text) == True
        
        # Test human text
        human_text = "I think this is a great idea"
        assert self.preprocessor.detect_bot_phrases(human_text) == False
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction."""
        text = "This is a test sentence. It has multiple sentences!"
        features = self.preprocessor.extract_linguistic_features(text)
        
        assert 'length' in features
        assert 'word_count' in features
        assert 'sentence_count' in features
        assert 'burstiness' in features
        assert 'lexical_diversity' in features
        assert features['sentence_count'] == 2
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        comments = [
            "This is a test comment",
            "Another comment with **markdown**",
            "Third comment with https://example.com"
        ]
        
        result = self.preprocessor.batch_preprocess(comments)
        
        assert len(result['cleaned_comments']) == 3
        assert len(result['features']) == 3
        assert len(result['bot_phrase_flags']) == 3
        assert "[URL]" in result['cleaned_comments'][2]
    
    def test_smart_sampling(self):
        """Test smart sampling strategy."""
        comments = [f"Comment {i}" for i in range(100)]
        sampled = self.preprocessor.smart_sampling(comments, sample_size=25)
        
        assert len(sampled) == 25
        # Should include recent comments
        assert "Comment 0" in sampled
        # Should include oldest comments
        assert "Comment 99" in sampled


class TestFastDetector:
    """Test fast detector."""
    
    @patch('models.detectors.fast_detector.pipeline')
    def test_fast_detector_initialization(self, mock_pipeline):
        """Test fast detector initialization."""
        mock_pipeline.return_value = Mock()
        
        detector = FastDetector()
        assert detector.model_name is not None
        assert detector.preprocessor is not None
    
    @patch('models.detectors.fast_detector.pipeline')
    def test_predict_single(self, mock_pipeline):
        """Test single prediction."""
        # Mock pipeline response
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value = [[
            {'label': 'LABEL_0', 'score': 0.8},
            {'label': 'LABEL_1', 'score': 0.2}
        ]]
        
        detector = FastDetector()
        result = detector.predict_single("Test text")
        
        assert 'bot_probability' in result
        assert 'confidence' in result
        assert isinstance(result['bot_probability'], float)
    
    @patch('models.detectors.fast_detector.pipeline')
    def test_analyze_user_comments(self, mock_pipeline):
        """Test user comment analysis."""
        # Mock pipeline response
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value = [[
            {'label': 'LABEL_0', 'score': 0.8},
            {'label': 'LABEL_1', 'score': 0.2}
        ]]
        
        detector = FastDetector()
        comments = ["Comment 1", "Comment 2", "Comment 3"]
        result = detector.analyze_user_comments(comments)
        
        assert 'bot_score' in result
        assert 'confidence' in result
        assert 'processing_time_ms' in result
        assert 'should_skip_deep' in result


class TestStatisticalAnalyzer:
    """Test statistical analyzer."""
    
    @patch('models.detectors.statistical_analyzer.PerplexityCalculator')
    @patch('models.detectors.statistical_analyzer.BPCCalculator')
    def test_statistical_analyzer_initialization(self, mock_bpc, mock_perplexity):
        """Test statistical analyzer initialization."""
        mock_perplexity.return_value = Mock()
        mock_bpc.return_value = Mock()
        
        analyzer = StatisticalAnalyzer()
        assert analyzer.preprocessor is not None
    
    def test_analyze_linguistic_features(self):
        """Test linguistic feature analysis."""
        analyzer = StatisticalAnalyzer()
        texts = ["This is a test sentence.", "Another test sentence!"]
        
        result = analyzer.analyze_linguistic_features(texts)
        
        assert 'bot_signal' in result
        assert 'burstiness' in result
        assert 'lexical_diversity' in result
        assert isinstance(result['bot_signal'], float)
    
    def test_get_available_analyzers(self):
        """Test getting available analyzers."""
        analyzer = StatisticalAnalyzer()
        analyzers = analyzer.get_available_analyzers()
        
        assert isinstance(analyzers, list)
        assert 'linguistic_features' in analyzers


class TestEnsembleScorer:
    """Test ensemble scorer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scorer = EnsembleScorer()
    
    def test_ensemble_scorer_initialization(self):
        """Test ensemble scorer initialization."""
        assert self.scorer.weights is not None
        assert self.scorer.threshold is not None
    
    def test_combine_fast_and_deep(self):
        """Test combining fast and deep results."""
        fast_result = AnalysisResult(
            stage="fast_screening",
            bot_score=70.0,
            confidence=80.0,
            processing_time_ms=100.0
        )
        
        deep_result = AnalysisResult(
            stage="deep_analysis",
            bot_score=75.0,
            confidence=85.0,
            processing_time_ms=200.0
        )
        
        result = self.scorer.combine_fast_and_deep(fast_result, deep_result)
        
        assert 'bot_score' in result
        assert 'confidence' in result
        assert 'stage' in result
        assert result['stage'] == 'fast_and_deep'
    
    def test_combine_fast_only(self):
        """Test combining fast results only."""
        fast_result = AnalysisResult(
            stage="fast_screening",
            bot_score=70.0,
            confidence=80.0,
            processing_time_ms=100.0
        )
        
        result = self.scorer.combine_fast_and_deep(fast_result, None)
        
        assert result['bot_score'] == 70.0
        assert result['stage'] == 'fast_only'
    
    def test_should_skip_deep_analysis(self):
        """Test deep analysis skip logic."""
        # High confidence bot result
        high_bot_result = AnalysisResult(
            stage="fast_screening",
            bot_score=90.0,  # Above threshold
            confidence=85.0,
            processing_time_ms=100.0,
            should_skip_next=True
        )
        
        assert self.scorer.should_skip_deep_analysis(high_bot_result) == True
        
        # Low confidence result
        low_result = AnalysisResult(
            stage="fast_screening",
            bot_score=50.0,  # Below threshold
            confidence=60.0,
            processing_time_ms=100.0,
            should_skip_next=False
        )
        
        assert self.scorer.should_skip_deep_analysis(low_result) == False


class TestBotDetectionAPI:
    """Test bot detection API."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.api = BotDetectionAPI()
    
    def test_api_initialization(self):
        """Test API initialization."""
        assert self.api.app is not None
        assert self.api._initialized == False
    
    def test_analysis_request_validation(self):
        """Test analysis request validation."""
        # Valid request
        valid_request = AnalysisRequest(
            user_id="test_user",
            comments=["Comment 1", "Comment 2"]
        )
        assert valid_request.user_id == "test_user"
        assert len(valid_request.comments) == 2
        
        # Request with options
        request_with_options = AnalysisRequest(
            user_id="test_user",
            comments=["Comment 1"],
            options=AnalysisOptions(fast_only=True)
        )
        assert request_with_options.options.fast_only == True


class TestMetrics:
    """Test metrics calculations."""
    
    def test_bpc_calculator(self):
        """Test BPC calculator."""
        calculator = BPCCalculator()
        
        # Test simple text
        text = "Hello world"
        bpc = calculator.calculate_bpc(text)
        assert isinstance(bpc, float)
        assert bpc > 0
        
        # Test empty text
        empty_bpc = calculator.calculate_bpc("")
        assert empty_bpc == 0.0
    
    def test_batch_bpc_calculation(self):
        """Test batch BPC calculation."""
        calculator = BPCCalculator()
        texts = ["Hello", "World", "Test"]
        
        bpc_scores = calculator.calculate_batch_bpc(texts)
        assert len(bpc_scores) == 3
        assert all(isinstance(score, float) for score in bpc_scores)


# Integration tests
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @patch('models.detectors.fast_detector.pipeline')
    @patch('models.detectors.deep_detector.pipeline')
    def test_full_pipeline_mock(self, mock_deep_pipeline, mock_fast_pipeline):
        """Test full pipeline with mocked models."""
        # Mock pipeline responses
        mock_fast_pipeline.return_value = Mock()
        mock_fast_pipeline.return_value.return_value = [[
            {'label': 'LABEL_0', 'score': 0.8},
            {'label': 'LABEL_1', 'score': 0.2}
        ]]
        
        mock_deep_pipeline.return_value = Mock()
        mock_deep_pipeline.return_value.return_value = [[
            {'label': 'LABEL_0', 'score': 0.7},
            {'label': 'LABEL_1', 'score': 0.3}
        ]]
        
        # Test preprocessing
        preprocessor = RedditPreprocessor()
        comments = ["This is a test comment", "Another test comment"]
        preprocessed = preprocessor.batch_preprocess(comments)
        
        assert len(preprocessed['cleaned_comments']) == 2
        
        # Test fast detector
        fast_detector = FastDetector()
        fast_result = fast_detector.analyze_user_comments(comments)
        
        assert 'bot_score' in fast_result
        assert 'should_skip_deep' in fast_result
        
        # Test ensemble scoring
        scorer = EnsembleScorer()
        fast_analysis_result = AnalysisResult(
            stage="fast_screening",
            bot_score=fast_result['bot_score'],
            confidence=fast_result['confidence'],
            processing_time_ms=fast_result['processing_time_ms'],
            should_skip_next=fast_result['should_skip_deep']
        )
        
        result = scorer.combine_fast_and_deep(fast_analysis_result, None)
        
        assert 'bot_score' in result
        assert 'confidence' in result
        assert 'is_likely_bot' in result


# Performance tests
class TestPerformance:
    """Performance tests."""
    
    def test_preprocessing_performance(self):
        """Test preprocessing performance."""
        preprocessor = RedditPreprocessor()
        comments = [f"This is test comment number {i}" for i in range(100)]
        
        import time
        start_time = time.time()
        result = preprocessor.batch_preprocess(comments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should process 100 comments in under 1 second
        assert len(result['cleaned_comments']) == 100
    
    def test_smart_sampling_performance(self):
        """Test smart sampling performance."""
        preprocessor = RedditPreprocessor()
        comments = [f"Comment {i}" for i in range(1000)]
        
        import time
        start_time = time.time()
        sampled = preprocessor.smart_sampling(comments, sample_size=50)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 0.1  # Should sample in under 0.1 seconds
        assert len(sampled) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
