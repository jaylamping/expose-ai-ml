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

## Examples

Run the example to see the library in action:

```bash
python example.py
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
