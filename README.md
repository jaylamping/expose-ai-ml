# Expose AI ML Library

A flexible Python library for machine learning with support for PyTorch, ONNX, and LangChain, designed for local development with easy Google Cloud deployment.

## Features

- **PyTorch Support**: Train and run PyTorch models locally
- **ONNX Integration**: Convert PyTorch models to ONNX for optimized inference
- **LangChain Integration**: Build LLM applications with OpenAI, Anthropic, and Google models
- **Google Cloud Ready**: Easy deployment to Google Cloud AI Platform
- **Device Management**: Automatic GPU/CPU device selection
- **Simple API**: Clean, intuitive interface for all ML operations

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jaylamping/expose-ai-ml.git
cd expose-ai-ml

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install -r requirements.txt[gpu]
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
# ML Framework Settings
DEVICE=auto
MODEL_CACHE_DIR=./models/cache

# Google Cloud Settings (for future deployment)
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# LangChain API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

### 3. Basic Usage

```python
from core.framework import MLFramework
from llm.chain import LLMChain

# Initialize ML Framework
ml_framework = MLFramework()

# Load a PyTorch model
model = ml_framework.load_pytorch_model("model.pth", "my_model")

# Convert to ONNX for faster inference
ml_framework.convert_pytorch_to_onnx(
    model,
    input_shape=(1, 784),
    output_path="model.onnx"
)

# Load ONNX model
ml_framework.load_onnx_model("model.onnx", "my_onnx_model")

# Make predictions
input_data = torch.randn(1, 784)
pytorch_result = ml_framework.predict_pytorch("my_model", input_data)
onnx_result = ml_framework.predict_onnx("my_onnx_model", input_data)

# Use LangChain for LLM applications
llm_chain = LLMChain(provider="openai")
chain = llm_chain.create_chain(
    template="Summarize this text: {text}",
    input_variables=["text"]
)
result = llm_chain.predict("Your text here", chain_name="default")
```

## Runtime Recommendations

### For Your Use Case (Local → Google Cloud):

**Primary: PyTorch**

- Best for development and training
- Excellent debugging capabilities
- Native Google Cloud support
- Easy to export to ONNX

**Secondary: ONNX**

- Use for production inference
- Better performance and smaller models
- Cross-platform compatibility
- Easy deployment to edge devices

**Hybrid Approach:**

1. Develop and train with PyTorch
2. Export to ONNX for production
3. Use LangChain for LLM applications
4. Deploy everything to Google Cloud

## Project Structure

```
expose-ai-ml/
├── core/                 # Core ML framework
│   ├── framework.py     # Main ML framework class
│   └── device_manager.py # Device management
├── llm/                 # LangChain integration
│   ├── chain.py        # LLM chain wrapper
│   ├── embeddings.py   # Embedding management
│   └── prompts.py      # Pre-built prompts
├── config/             # Configuration
│   └── settings.py     # Settings management
├── api/                # API endpoints (future)
├── models/             # Model storage
├── example.py          # Usage example
├── requirements.txt    # Dependencies
└── setup.py           # Package setup
```

## Bot Detection API

The library now includes a sophisticated multi-stage bot detection system for analyzing social media content.

### Quick Start with Bot Detection

```python
from api.bot_detection import BotDetectionAPI

# Initialize the API
api = BotDetectionAPI()

# Start the server
api.run(host="0.0.0.0", port=8000)
```

### API Usage

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

### Multi-Stage Detection Pipeline

The bot detection system uses a sophisticated multi-stage approach:

1. **Fast Screening** (200-300ms): Quick analysis using lightweight models
2. **Deep Analysis** (1-2s): High-accuracy analysis using DeBERTa-v3-base
3. **Statistical Analysis** (500ms-1s): Perplexity, BPC, sentiment consistency, embedding similarity
4. **Ensemble Scoring**: Weighted combination of all signals

### Features

- **Reddit-specific preprocessing**: Handles markdown, URLs, mentions, subreddits
- **Smart sampling**: Efficiently processes large comment sets
- **Context awareness**: Uses parent comment context for better accuracy
- **Multiple signals**: 10+ different analysis methods
- **Confidence scoring**: Provides confidence levels for all predictions
- **Batch processing**: Optimized for processing multiple comments
- **ONNX optimization**: Fast inference for production deployment

### Training Custom Models

```python
from models.training.train_detector import BotDetectionTrainer

# Initialize trainer
trainer = BotDetectionTrainer(
    model_name="microsoft/deberta-v3-base",
    output_dir="./models/trained"
)

# Run full training pipeline
model_path = trainer.run_full_training_pipeline()
print(f"Model saved to: {model_path}")
```

## Examples

Run the example to see the library in action:

```bash
python example.py
```

Run the bot detection API:

```bash
python api/bot_detection.py
```

## Google Cloud Deployment

When ready to deploy to Google Cloud:

1. Set up your GCP project and credentials
2. Use the built-in Google Cloud integration
3. Deploy models to AI Platform
4. Scale with Cloud Run or Kubernetes

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
