# Bot Detection API

A sophisticated multi-stage bot detection system for analyzing social media content, with a focus on Reddit comment analysis. Built with PyTorch, Transformers, and FastAPI for high-performance inference.

## Features

- **Multi-Stage Detection Pipeline**: Fast screening → Deep analysis → Statistical analysis → Ensemble scoring
- **Reddit-Optimized**: Specialized preprocessing for Reddit comments, markdown, URLs, and mentions
- **High-Performance Models**: Uses RoBERTa and DeBERTa-v3-base for accurate bot detection
- **Statistical Analysis**: Perplexity, BPC, sentiment consistency, embedding similarity, and more
- **FastAPI Integration**: Production-ready REST API with automatic documentation
- **ONNX Optimization**: Fast inference with ONNX Runtime for production deployment
- **Smart Sampling**: Efficiently processes large comment sets with intelligent sampling
- **Context Awareness**: Uses parent comment context for improved accuracy
- **Confidence Scoring**: Provides confidence levels for all predictions

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jaylamping/expose-ai-ml.git
cd expose-ai-ml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Development mode (recommended for testing)
python run_dev.py

# Or run directly
python api/bot_detection.py
```

The API will be available at:

- **Local**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Test the API

```bash
# Run the test script
python test_api.py
```

### 4. Basic Usage

```python
import requests

# Analyze a user's comments
response = requests.post("http://localhost:8000/api/v1/analyze-user", json={
    "user_id": "reddit_username",
    "comments": [
        "This is a great post!",
        "I completely agree with your point.",
        "Thanks for sharing this information."
    ],
    "options": {
        "fast_only": False,
        "include_breakdown": True,
        "use_context": True
    }
})

result = response.json()
print(f"Bot Score: {result['bot_score']}%")
print(f"Confidence: {result['confidence']}%")
print(f"Is Bot: {result['is_likely_bot']}")
```

## Project Structure

```
expose-ai-ml/
├── api/                    # Bot Detection API
│   └── bot_detection.py   # FastAPI server and endpoints
├── models/                 # Bot detection models
│   ├── detectors/         # Detection algorithms
│   │   ├── fast_detector.py      # Fast screening (RoBERTa)
│   │   ├── deep_detector.py      # Deep analysis (DeBERTa-v3)
│   │   └── statistical_analyzer.py # Statistical metrics
│   ├── ensemble.py        # Ensemble scoring system
│   └── training/          # Model training utilities
├── utils/                  # Utilities
│   ├── preprocessing.py   # Reddit comment preprocessing
│   └── metrics.py         # Analysis metrics and tools
├── config/                # Configuration
│   └── settings.py        # Application settings
├── core/                  # Core framework (legacy)
│   ├── framework.py       # ML framework (basic implementation)
│   └── device_manager.py  # Device management
├── llm/                   # LangChain integration (basic)
│   ├── chain.py          # LLM chain wrapper
│   ├── embeddings.py     # Embedding management
│   └── prompts.py        # Pre-built prompts
├── run_dev.py            # Development server launcher
├── test_api.py           # API testing script
├── requirements.txt      # Dependencies
└── setup.py             # Package setup
```

## API Reference

### Request Format

**Endpoint**: `POST /api/v1/analyze-user`

**Request Example**:

```json
{
  "user_id": "reddit_username",
  "comments": [
    "This is a great post!",
    "I completely agree with your point.",
    "Thanks for sharing this information."
  ],
  "parent_contexts": [
    { "comment_id": "c1", "parent": "Original post text" },
    { "comment_id": "c2", "parent": null }
  ],
  "options": {
    "fast_only": false,
    "include_breakdown": true,
    "use_context": true,
    "force_full_analysis": false
  }
}
```

**Response Example**:

```json
{
  "user_id": "reddit_username",
  "bot_score": 73.5,
  "confidence": 85.2,
  "is_likely_bot": true,
  "stage": "full_ensemble",
  "processing_time_ms": 1834,
  "comments_analyzed": 25,
  "total_comments": 3,
  "breakdown": {
    "fast_model": 70.0,
    "deep_model": 78.0,
    "perplexity": 72.0,
    "bpc": 74.0,
    "sentiment_consistency": 80.0,
    "embedding_similarity": 75.0,
    "zero_shot": 65.0,
    "burstiness": 70.0
  },
  "explanation": "Analysis using full_ensemble with high confidence (85.2%). Result: likely bot (73.5% bot score)."
}
```

### Detection Pipeline

The system uses a sophisticated multi-stage approach:

1. **Fast Screening** (200-300ms): Quick analysis using lightweight models
2. **Deep Analysis** (1-2s): High-accuracy analysis using DeBERTa-v3-base
3. **Statistical Analysis** (500ms-1s): Perplexity, BPC, sentiment consistency, embedding similarity
4. **Ensemble Scoring**: Weighted combination of all signals

### Advanced Features

- **Reddit-specific preprocessing**: Handles markdown, URLs, mentions, subreddits
- **Smart sampling**: Efficiently processes large comment sets
- **Context awareness**: Uses parent comment context for better accuracy
- **Multiple signals**: 10+ different analysis methods
- **Confidence scoring**: Provides confidence levels for all predictions
- **Batch processing**: Optimized for processing multiple comments
- **ONNX optimization**: Fast inference for production deployment

## Examples

### Start the Development Server

```bash
# Start with development settings (recommended)
python run_dev.py

# Or start directly
python api/bot_detection.py
```

### Test the API

```bash
# Run comprehensive API tests
python test_api.py
```

### API Endpoints

- **Health Check**: `GET /health`
- **Root Info**: `GET /`
- **Analyze User**: `POST /api/v1/analyze-user`
- **Models Info**: `GET /api/v1/models/info`
- **API Documentation**: `GET /docs` (Swagger UI)

## Performance & Deployment

### Performance Characteristics

- **Fast Mode**: 200-300ms per analysis (lightweight models)
- **Full Analysis**: 1-2 seconds per analysis (comprehensive pipeline)
- **Memory Usage**: ~2-4GB RAM (depending on models loaded)
- **GPU Support**: Automatic GPU detection and utilization

### Production Deployment

The API is built with FastAPI and can be deployed using:

- **Docker**: Containerize the application
- **Cloud Run**: Serverless deployment on Google Cloud
- **Kubernetes**: Scalable container orchestration
- **Traditional VPS**: Direct deployment on virtual servers

### Environment Variables

```env
# Device configuration
DEVICE=auto  # or cuda, cpu

# Model cache directory
MODEL_CACHE_DIR=./models/cache

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact Joey Lamping.
