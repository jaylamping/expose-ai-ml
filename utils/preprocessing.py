"""
Reddit comment preprocessing and batch processing utilities.
"""
import re
import string
from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]


class RedditPreprocessor:
    """Preprocessor for Reddit comments with batch processing capabilities."""
    
    def __init__(self, max_length: int = 512):
        """
        Initialize the preprocessor.
        
        Args:
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
        
        # Common Reddit patterns
        self.reddit_patterns = {
            'url': re.compile(r'https?://\S+'),
            'user_mention': re.compile(r'u/\w+'),
            'subreddit': re.compile(r'r/\w+'),
            'bold_markdown': re.compile(r'\*\*(.+?)\*\*'),
            'italic_markdown': re.compile(r'\*(.+?)\*'),
            'link_markdown': re.compile(r'\[(.+?)\]\(.+?\)'),
            'code_block': re.compile(r'```.*?```', re.DOTALL),
            'inline_code': re.compile(r'`([^`]+)`'),
            'quote_block': re.compile(r'^>.*$', re.MULTILINE),
            'spoiler': re.compile(r'>!.*?!<', re.DOTALL),
            'strikethrough': re.compile(r'~~(.+?)~~'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n\s*\n'),
        }
        
        # Bot-specific patterns
        self.bot_phrases = [
            r'\b(?:as an ai|i am an ai|i\'m an ai)\b',
            r'\b(?:i don\'t have (?:personal )?opinions?)\b',
            r'\b(?:i cannot (?:have|form) opinions?)\b',
            r'\b(?:i am not capable of)\b',
            r'\b(?:i am designed to)\b',
            r'\b(?:i am programmed to)\b',
            r'\b(?:i am a language model)\b',
            r'\b(?:i am chatgpt|i am gpt)\b',
            r'\b(?:i am claude|i am anthropic)\b',
            r'\b(?:i am an assistant)\b',
        ]
        
        self.bot_phrase_pattern = re.compile('|'.join(self.bot_phrases), re.IGNORECASE)
    
    def clean_reddit_text(self, text: str) -> str:
        """
        Clean Reddit-specific formatting and markdown.
        
        Args:
            text: Raw Reddit comment text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = self.reddit_patterns['url'].sub('[URL]', text)
        
        # Normalize mentions and subreddits
        text = self.reddit_patterns['user_mention'].sub('[USER]', text)
        text = self.reddit_patterns['subreddit'].sub('[SUBREDDIT]', text)
        
        # Remove markdown formatting
        text = self.reddit_patterns['bold_markdown'].sub(r'\1', text)
        text = self.reddit_patterns['italic_markdown'].sub(r'\1', text)
        text = self.reddit_patterns['link_markdown'].sub(r'\1', text)
        text = self.reddit_patterns['strikethrough'].sub(r'\1', text)
        
        # Remove code blocks and inline code
        text = self.reddit_patterns['code_block'].sub('[CODE]', text)
        text = self.reddit_patterns['inline_code'].sub(r'\1', text)
        
        # Remove quote blocks and spoilers
        text = self.reddit_patterns['quote_block'].sub('', text)
        text = self.reddit_patterns['spoiler'].sub('[SPOILER]', text)
        
        # Normalize whitespace
        text = self.reddit_patterns['multiple_spaces'].sub(' ', text)
        text = self.reddit_patterns['multiple_newlines'].sub('\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def detect_bot_phrases(self, text: str) -> bool:
        """
        Detect common bot phrases in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if bot phrases detected
        """
        return bool(self.bot_phrase_pattern.search(text))
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of linguistic features
        """
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'burstiness': 0,
                'lexical_diversity': 0,
                'emoji_count': 0,
                'em_dash_count': 0,
                'punctuation_ratio': 0
            }
        
        # Basic counts
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Emoji detection
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        emoji_count = len(emoji_pattern.findall(text))
        
        # Em dash detection
        em_dash_count = text.count('—')
        
        # Punctuation ratio
        punctuation_count = sum(1 for c in text if c in string.punctuation)
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        # Sentence length variation (burstiness)
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            burstiness = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
            avg_sentence_length = np.mean(sentence_lengths)
        else:
            burstiness = 0
            avg_sentence_length = 0
        
        # Lexical diversity (type-token ratio)
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        return {
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'burstiness': burstiness,
            'lexical_diversity': lexical_diversity,
            'emoji_count': emoji_count,
            'em_dash_count': em_dash_count,
            'punctuation_ratio': punctuation_ratio
        }
    
    def batch_preprocess(self, comments: List[str], include_features: bool = True) -> Dict:
        """
        Preprocess a batch of comments.
        
        Args:
            comments: List of comment texts
            include_features: Whether to extract linguistic features
            
        Returns:
            Dictionary with cleaned comments and optional features
        """
        cleaned_comments = []
        features = []
        bot_phrase_flags = []
        
        for comment in comments:
            # Clean the text
            cleaned = self.clean_reddit_text(comment)
            cleaned_comments.append(cleaned)
            
            # Extract features if requested
            if include_features:
                features.append(self.extract_linguistic_features(cleaned))
                bot_phrase_flags.append(self.detect_bot_phrases(cleaned))
        
        result = {
            'cleaned_comments': cleaned_comments,
            'original_comments': comments
        }
        
        if include_features:
            result['features'] = features
            result['bot_phrase_flags'] = bot_phrase_flags
        
        return result
    
    def prepare_for_tokenization(self, comments: List[str], tokenizer: AutoTokenizer) -> Dict:
        """
        Prepare comments for tokenization with proper batching.
        
        Args:
            comments: List of comment texts
            tokenizer: HuggingFace tokenizer
            
        Returns:
            Tokenized inputs ready for model inference
        """
        # Clean comments first
        preprocessed = self.batch_preprocess(comments, include_features=False)
        cleaned_comments = preprocessed['cleaned_comments']
        
        # Tokenize with padding and truncation
        inputs = tokenizer(
            cleaned_comments,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'cleaned_comments': cleaned_comments
        }
    
    def smart_sampling(self, comments: List[str], sample_size: int = 25) -> List[str]:
        """
        Smart sampling strategy for large comment sets.
        
        Args:
            comments: List of all comments
            sample_size: Number of comments to sample
            
        Returns:
            Sampled comments
        """
        if len(comments) <= sample_size:
            return comments
        
        # Strategy: recent + random + oldest
        recent_count = min(10, sample_size // 3)
        oldest_count = min(5, sample_size // 6)
        random_count = sample_size - recent_count - oldest_count
        
        # Get recent comments (first in list)
        recent = comments[:recent_count]
        
        # Get oldest comments (last in list)
        oldest = comments[-oldest_count:] if oldest_count > 0 else []
        
        # Get random sample from middle
        middle_comments = comments[recent_count:-oldest_count] if oldest_count > 0 else comments[recent_count:]
        if middle_comments and random_count > 0:
            random_indices = np.random.choice(
                len(middle_comments), 
                size=min(random_count, len(middle_comments)), 
                replace=False
            )
            random_sample = [middle_comments[i] for i in random_indices]
        else:
            random_sample = []
        
        return recent + random_sample + oldest
    
    def create_context_pairs(self, comments: List[str], parent_contexts: List[Dict]) -> List[str]:
        """
        Create comment-context pairs for better analysis.
        
        Args:
            comments: List of comment texts
            parent_contexts: List of parent context dictionaries
            
        Returns:
            List of formatted comment-context pairs
        """
        formatted_pairs = []
        
        for i, comment in enumerate(comments):
            if i < len(parent_contexts) and parent_contexts[i].get('parent'):
                # Combine comment with parent context
                parent = parent_contexts[i]['parent']
                formatted = f"[CONTEXT] {parent} [COMMENT] {comment}"
            else:
                # No context available
                formatted = f"[COMMENT] {comment}"
            
            formatted_pairs.append(formatted)
        
        return formatted_pairs


def aggregate_linguistic_features(features: List[Dict]) -> Dict[str, float]:
    """
    Aggregate linguistic features across multiple comments.
    
    Args:
        features: List of feature dictionaries
        
    Returns:
        Aggregated features
    """
    if not features:
        return {}
    
    # Aggregate numerical features
    aggregated = {}
    for key in features[0].keys():
        values = [f[key] for f in features if key in f]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
    
    return aggregated
