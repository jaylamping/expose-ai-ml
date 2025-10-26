"""
Configuration settings for the ML library.
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field


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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
