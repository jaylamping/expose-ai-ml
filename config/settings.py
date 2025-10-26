"""
Configuration settings for the ML library.
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # ML Framework Settings
    device: str = Field(default="auto", description="Device to use for ML models (cpu, cuda, auto)")
    model_cache_dir: str = Field(default="./models/cache", description="Directory to cache models")
    
    # Google Cloud Settings
    gcp_project_id: Optional[str] = Field(default=None, description="Google Cloud Project ID")
    gcp_region: str = Field(default="us-central1", description="Google Cloud region")
    gcp_credentials_path: Optional[str] = Field(default=None, description="Path to GCP credentials JSON")
    
    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_timeout: int = Field(default=120, description="Ollama request timeout in seconds")
    
    # Model Settings
    default_llm_model: str = Field(default="llama2", description="Default Ollama model")
    default_embedding_model: str = Field(default="nomic-embed-text", description="Default embedding model")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_debug: bool = Field(default=False, description="API debug mode")
    
    # Bot Detection Settings
    bot_detection_model_cache: str = Field(default="./models/cache/bot_detection", description="Bot detection model cache directory")
    fast_model_name: str = Field(default="roberta-base-openai-detector", description="Fast screening model")
    deep_model_name: str = Field(default="microsoft/deberta-v3-base", description="Deep analysis model")
    perplexity_model_name: str = Field(default="gpt2", description="Model for perplexity calculation")
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Sentence embedding model")
    
    # Bot Detection Thresholds
    fast_threshold_low: float = Field(default=0.15, description="Fast model low threshold (skip deep analysis)")
    fast_threshold_high: float = Field(default=0.85, description="Fast model high threshold (skip deep analysis)")
    bot_threshold: float = Field(default=0.7, description="Final bot classification threshold")
    
    # Ensemble Weights
    ensemble_weights: dict = Field(default={
        "fast_model": 0.25,
        "deep_model": 0.30,
        "perplexity": 0.10,
        "bpc": 0.08,
        "sentiment_consistency": 0.12,
        "embedding_similarity": 0.08,
        "zero_shot": 0.05,
        "burstiness": 0.02
    }, description="Ensemble scoring weights")
    
    # Processing Settings
    max_comments_per_request: int = Field(default=100, description="Maximum comments to process per request")
    batch_size: int = Field(default=32, description="Batch size for model inference")
    max_sequence_length: int = Field(default=512, description="Maximum token sequence length")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


# Global settings instance
settings = Settings()
