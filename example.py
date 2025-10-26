"""
Simple example showing how to use the ML library with PyTorch, ONNX, and LangChain.
"""
import torch
import numpy as np
from core.framework import MLFramework
from llm.chain import LLMChain
from config.settings import settings

def main():
    """Main example function."""
    print("🚀 Expose AI ML Library Example")
    print("=" * 40)
    
    # Initialize ML Framework
    print("\n1. Initializing ML Framework...")
    ml_framework = MLFramework()
    print(f"   Device: {ml_framework.device_manager.get_device()}")
    
    # Example: Create a simple PyTorch model
    print("\n2. Creating a simple PyTorch model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Convert to ONNX
    print("\n3. Converting PyTorch model to ONNX...")
    onnx_path = "example_model.onnx"
    ml_framework.convert_pytorch_to_onnx(
        model, 
        input_shape=(1, 10), 
        output_path=onnx_path
    )
    print(f"   ONNX model saved to: {onnx_path}")
    
    # Load ONNX model
    print("\n4. Loading ONNX model...")
    ml_framework.load_onnx_model(onnx_path, "example_model")
    
    # Make predictions
    print("\n5. Making predictions...")
    test_input = torch.randn(1, 10)
    
    # PyTorch prediction
    pytorch_pred = ml_framework.predict_pytorch("example_model", test_input)
    print(f"   PyTorch prediction: {pytorch_pred.item():.4f}")
    
    # ONNX prediction
    onnx_pred = ml_framework.predict_onnx("example_model", test_input)
    print(f"   ONNX prediction: {onnx_pred[0][0]:.4f}")
    
    # LangChain with Ollama Example
    print("\n6. LangChain with Ollama Example...")
    try:
        # Create a simple LLM chain with Ollama
        llm_chain = LLMChain(model_name="llama2", temperature=0.7)
        
        # Create a simple prompt chain
        chain = llm_chain.create_chain(
            template="What is the capital of {country}?",
            input_variables=["country"]
        )
        
        # Make a prediction
        result = llm_chain.predict("France", chain_name="default")
        print(f"   LLM Response: {result}")
        
    except Exception as e:
        print(f"   LangChain example skipped: {e}")
        print("   (Make sure Ollama is running and llama2 model is installed)")
    
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("- Install Ollama: https://ollama.ai/")
    print("- Pull models: ollama pull llama2")
    print("- Load your own Hugging Face models")
    print("- Deploy to Google Cloud when ready")

if __name__ == "__main__":
    main()
