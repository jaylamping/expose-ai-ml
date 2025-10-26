"""
ONNX conversion utilities for optimizing model inference speed.
"""
import os
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from config.settings import settings


class ONNXConverter:
    """Converter for transforming PyTorch models to ONNX format."""
    
    def __init__(self, output_dir: str = "./models/onnx"):
        """
        Initialize the ONNX converter.
        
        Args:
            output_dir: Directory to save ONNX models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_fast_detector(self, 
                            model_name: str = "roberta-base-openai-detector",
                            output_name: str = "fast_detector.onnx") -> str:
        """
        Convert fast detector model to ONNX.
        
        Args:
            model_name: Name of the model to convert
            output_name: Name for the output ONNX file
            
        Returns:
            Path to the converted ONNX model
        """
        print(f"🔄 Converting {model_name} to ONNX...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dummy input
        dummy_text = "This is a test sentence for ONNX conversion."
        dummy_input = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length
        )
        
        # Convert to ONNX
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Save tokenizer
        tokenizer_path = self.output_dir / f"{output_name.replace('.onnx', '_tokenizer')}"
        tokenizer.save_pretrained(tokenizer_path)
        
        print(f"✅ Fast detector converted to ONNX: {output_path}")
        return str(output_path)
    
    def convert_deep_detector(self, 
                            model_name: str = "microsoft/deberta-v3-base",
                            output_name: str = "deep_detector.onnx") -> str:
        """
        Convert deep detector model to ONNX.
        
        Args:
            model_name: Name of the model to convert
            output_name: Name for the output ONNX file
            
        Returns:
            Path to the converted ONNX model
        """
        print(f"🔄 Converting {model_name} to ONNX...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dummy input
        dummy_text = "This is a test sentence for ONNX conversion."
        dummy_input = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length
        )
        
        # Convert to ONNX
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Save tokenizer
        tokenizer_path = self.output_dir / f"{output_name.replace('.onnx', '_tokenizer')}"
        tokenizer.save_pretrained(tokenizer_path)
        
        print(f"✅ Deep detector converted to ONNX: {output_path}")
        return str(output_path)
    
    def convert_perplexity_model(self, 
                               model_name: str = "gpt2",
                               output_name: str = "perplexity_model.onnx") -> str:
        """
        Convert perplexity model to ONNX.
        
        Args:
            model_name: Name of the model to convert
            output_name: Name for the output ONNX file
            
        Returns:
            Path to the converted ONNX model
        """
        print(f"🔄 Converting {model_name} to ONNX...")
        
        from transformers import AutoModelForCausalLM
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dummy input
        dummy_text = "This is a test sentence for ONNX conversion."
        dummy_input = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length
        )
        
        # Convert to ONNX
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            dummy_input["input_ids"],
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Save tokenizer
        tokenizer_path = self.output_dir / f"{output_name.replace('.onnx', '_tokenizer')}"
        tokenizer.save_pretrained(tokenizer_path)
        
        print(f"✅ Perplexity model converted to ONNX: {output_path}")
        return str(output_path)
    
    def convert_all_models(self) -> Dict[str, str]:
        """
        Convert all models to ONNX format.
        
        Returns:
            Dictionary mapping model names to ONNX file paths
        """
        converted_models = {}
        
        try:
            # Convert fast detector
            fast_path = self.convert_fast_detector()
            converted_models["fast_detector"] = fast_path
        except Exception as e:
            print(f"❌ Failed to convert fast detector: {e}")
        
        try:
            # Convert deep detector
            deep_path = self.convert_deep_detector()
            converted_models["deep_detector"] = deep_path
        except Exception as e:
            print(f"❌ Failed to convert deep detector: {e}")
        
        try:
            # Convert perplexity model
            perplexity_path = self.convert_perplexity_model()
            converted_models["perplexity_model"] = perplexity_path
        except Exception as e:
            print(f"❌ Failed to convert perplexity model: {e}")
        
        print(f"🎉 ONNX conversion complete! Converted {len(converted_models)} models.")
        return converted_models
    
    def benchmark_onnx_vs_pytorch(self, 
                                 onnx_path: str, 
                                 pytorch_model, 
                                 tokenizer,
                                 test_texts: List[str],
                                 num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark ONNX vs PyTorch inference speed.
        
        Args:
            onnx_path: Path to ONNX model
            pytorch_model: PyTorch model
            tokenizer: Tokenizer
            test_texts: List of test texts
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        print(f"🏃 Benchmarking ONNX vs PyTorch...")
        
        # Load ONNX model
        onnx_session = ort.InferenceSession(onnx_path)
        
        # Prepare test data
        inputs = tokenizer(
            test_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length
        )
        
        # Benchmark PyTorch
        pytorch_times = []
        pytorch_model.eval()
        
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            else:
                import time
                start = time.time()
            
            with torch.no_grad():
                _ = pytorch_model(**inputs)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                pytorch_times.append(start_time.elapsed_time(end_time))
            else:
                pytorch_times.append((time.time() - start) * 1000)
        
        # Benchmark ONNX
        onnx_times = []
        input_ids_np = inputs["input_ids"].numpy()
        attention_mask_np = inputs["attention_mask"].numpy()
        
        for _ in range(num_runs):
            import time
            start = time.time()
            
            _ = onnx_session.run(
                None,
                {
                    "input_ids": input_ids_np,
                    "attention_mask": attention_mask_np
                }
            )
            
            onnx_times.append((time.time() - start) * 1000)
        
        # Calculate statistics
        pytorch_mean = np.mean(pytorch_times)
        pytorch_std = np.std(pytorch_times)
        onnx_mean = np.mean(onnx_times)
        onnx_std = np.std(onnx_times)
        
        speedup = pytorch_mean / onnx_mean if onnx_mean > 0 else 0
        
        results = {
            "pytorch_mean_ms": pytorch_mean,
            "pytorch_std_ms": pytorch_std,
            "onnx_mean_ms": onnx_mean,
            "onnx_std_ms": onnx_std,
            "speedup": speedup,
            "num_runs": num_runs,
            "batch_size": len(test_texts)
        }
        
        print(f"📊 Benchmark Results:")
        print(f"  PyTorch: {pytorch_mean:.2f} ± {pytorch_std:.2f} ms")
        print(f"  ONNX: {onnx_mean:.2f} ± {onnx_std:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        return results


class ONNXInference:
    """ONNX inference wrapper for optimized model inference."""
    
    def __init__(self, onnx_path: str, tokenizer_path: str):
        """
        Initialize ONNX inference.
        
        Args:
            onnx_path: Path to ONNX model
            tokenizer_path: Path to tokenizer
        """
        self.onnx_path = onnx_path
        self.tokenizer_path = tokenizer_path
        
        # Load ONNX session
        self.session = ort.InferenceSession(onnx_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Get input/output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions using ONNX model.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Predictions as numpy array
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length
        )
        
        # Prepare input dictionary
        input_dict = {}
        for input_name in self.input_names:
            if input_name == "input_ids":
                input_dict[input_name] = inputs["input_ids"]
            elif input_name == "attention_mask":
                input_dict[input_name] = inputs["attention_mask"]
        
        # Run inference
        outputs = self.session.run(self.output_names, input_dict)
        
        return outputs[0]  # Return first output (logits)
    
    def predict_single(self, text: str) -> float:
        """
        Make prediction for a single text.
        
        Args:
            text: Text to predict
            
        Returns:
            Bot probability score
        """
        predictions = self.predict([text])
        
        # Convert logits to probabilities
        probabilities = torch.softmax(torch.tensor(predictions), dim=-1)
        bot_probability = probabilities[0][1].item()  # Assuming class 1 is bot
        
        return bot_probability


def main():
    """Main function to convert all models to ONNX."""
    converter = ONNXConverter()
    converted_models = converter.convert_all_models()
    
    print("🎉 ONNX conversion completed!")
    print("Converted models:")
    for name, path in converted_models.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
