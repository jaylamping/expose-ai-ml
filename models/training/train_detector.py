"""
Training pipeline for fine-tuning DeBERTa on bot detection datasets.
"""
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config.settings import settings


class BotDetectionTrainer:
    """Trainer for fine-tuning DeBERTa on bot detection."""
    
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-base",
                 output_dir: str = "./models/trained",
                 cache_dir: str = None):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save trained model
            cache_dir: Directory to cache datasets
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir or settings.bot_detection_model_cache
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the base model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # Binary classification: human vs bot
            problem_type="single_label_classification"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model loaded: {self.model_name}")
    
    def load_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and prepare datasets for training.
        
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        print("📊 Loading datasets...")
        
        # Load multiple datasets and combine them
        datasets = []
        
        # 1. HC3 dataset (ChatGPT vs Human answers)
        try:
            print("Loading HC3 dataset...")
            hc3_dataset = load_dataset("Hello-SimpleAI/HC3", "all", cache_dir=self.cache_dir)
            
            # Process HC3 data
            hc3_processed = self._process_hc3_dataset(hc3_dataset)
            datasets.append(hc3_processed)
            print(f"✅ HC3 dataset loaded: {len(hc3_processed)} samples")
        except Exception as e:
            print(f"⚠️ Failed to load HC3 dataset: {e}")
        
        # 2. TruthfulQA dataset
        try:
            print("Loading TruthfulQA dataset...")
            truthfulqa_dataset = load_dataset("truthful_qa", "generation", cache_dir=self.cache_dir)
            
            # Process TruthfulQA data
            truthfulqa_processed = self._process_truthfulqa_dataset(truthfulqa_dataset)
            datasets.append(truthfulqa_processed)
            print(f"✅ TruthfulQA dataset loaded: {len(truthfulqa_processed)} samples")
        except Exception as e:
            print(f"⚠️ Failed to load TruthfulQA dataset: {e}")
        
        # 3. Synthetic dataset (if available)
        try:
            print("Loading synthetic bot dataset...")
            synthetic_dataset = self._create_synthetic_dataset()
            datasets.append(synthetic_dataset)
            print(f"✅ Synthetic dataset created: {len(synthetic_dataset)} samples")
        except Exception as e:
            print(f"⚠️ Failed to create synthetic dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Combine all datasets
        print("Combining datasets...")
        combined_dataset = self._combine_datasets(datasets)
        
        # Split into train/validation/test
        train_val_test = combined_dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=42)
        
        train_dataset = train_val_test['train']
        val_dataset = val_test['train']
        test_dataset = val_test['test']
        
        print(f"📊 Dataset splits:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def _process_hc3_dataset(self, dataset) -> List[Dict]:
        """Process HC3 dataset for training."""
        processed = []
        
        for split in ['train', 'validation']:
            if split in dataset:
                for item in dataset[split]:
                    # Human answers
                    for human_answer in item.get('human_answers', []):
                        processed.append({
                            'text': human_answer,
                            'label': 0,  # Human
                            'source': 'hc3_human'
                        })
                    
                    # ChatGPT answers
                    for chatgpt_answer in item.get('chatgpt_answers', []):
                        processed.append({
                            'text': chatgpt_answer,
                            'label': 1,  # Bot
                            'source': 'hc3_chatgpt'
                        })
        
        return processed
    
    def _process_truthfulqa_dataset(self, dataset) -> List[Dict]:
        """Process TruthfulQA dataset for training."""
        processed = []
        
        for item in dataset['validation']:
            # Best answer (human-like)
            if 'Best Answer' in item:
                processed.append({
                    'text': item['Best Answer'],
                    'label': 0,  # Human
                    'source': 'truthfulqa_human'
                })
            
            # Incorrect answers (AI-like)
            for incorrect in item.get('Incorrect Answers', []):
                processed.append({
                    'text': incorrect,
                    'label': 1,  # Bot
                    'source': 'truthfulqa_incorrect'
                })
        
        return processed
    
    def _create_synthetic_dataset(self) -> List[Dict]:
        """Create synthetic dataset with known bot patterns."""
        synthetic_data = []
        
        # Bot-like patterns
        bot_patterns = [
            "As an AI, I don't have personal opinions, but I can provide information about...",
            "I am designed to be helpful, harmless, and honest. Based on my training...",
            "I cannot provide personal opinions, but I can share factual information...",
            "As a language model, I don't have the ability to form personal beliefs...",
            "I am programmed to assist with information and answer questions...",
            "Based on my training data, I can provide the following information...",
            "I don't have personal experiences, but I can help you understand...",
            "As an AI assistant, my goal is to be helpful and informative...",
            "I cannot make subjective judgments, but I can present facts...",
            "My responses are generated based on patterns in my training data..."
        ]
        
        # Human-like patterns
        human_patterns = [
            "I think this is a really interesting question. From my experience...",
            "Honestly, I'm not sure about this, but I've heard that...",
            "This reminds me of something that happened to me once...",
            "I disagree with that approach. In my opinion...",
            "I've been thinking about this a lot lately, and I believe...",
            "From what I've seen, this doesn't usually work because...",
            "I'm not an expert, but I've noticed that...",
            "This is just my personal take, but I think...",
            "I've had mixed results with this approach...",
            "I'm not sure I understand the question completely, but..."
        ]
        
        # Add bot patterns
        for pattern in bot_patterns:
            synthetic_data.append({
                'text': pattern,
                'label': 1,  # Bot
                'source': 'synthetic_bot'
            })
        
        # Add human patterns
        for pattern in human_patterns:
            synthetic_data.append({
                'text': pattern,
                'label': 0,  # Human
                'source': 'synthetic_human'
            })
        
        return synthetic_data
    
    def _combine_datasets(self, datasets: List[List[Dict]]) -> Dataset:
        """Combine multiple datasets into one."""
        combined_data = []
        
        for dataset in datasets:
            combined_data.extend(dataset)
        
        # Shuffle the combined data
        np.random.shuffle(combined_data)
        
        return Dataset.from_list(combined_data)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=settings.max_sequence_length
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, 
              train_dataset: Dataset,
              val_dataset: Dataset,
              num_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500) -> Trainer:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            
        Returns:
            Trained trainer object
        """
        print("🚀 Starting training...")
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = self.tokenize_dataset(train_dataset)
        val_dataset = self.tokenize_dataset(val_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            seed=42
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        print("🔥 Training started...")
        self.trainer.train()
        
        print("✅ Training completed!")
        
        return self.trainer
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the trained model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Model must be trained before evaluation!")
        
        print("📊 Evaluating model...")
        
        # Tokenize test dataset
        test_dataset = self.tokenize_dataset(test_dataset)
        
        # Run evaluation
        eval_results = self.trainer.evaluate(test_dataset)
        
        print("📈 Evaluation Results:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return eval_results
    
    def save_model(self, model_name: str = None):
        """
        Save the trained model and tokenizer.
        
        Args:
            model_name: Name for the saved model
        """
        if not self.trainer:
            raise ValueError("Model must be trained before saving!")
        
        model_name = model_name or f"deberta-bot-detector-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = self.output_dir / model_name
        
        print(f"💾 Saving model to {save_path}")
        
        # Save model and tokenizer
        self.trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "trained_model_name": model_name,
            "save_path": str(save_path),
            "training_date": datetime.now().isoformat(),
            "num_labels": 2,
            "max_sequence_length": settings.max_sequence_length
        }
        
        with open(save_path / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"✅ Model saved to {save_path}")
        
        return str(save_path)
    
    def run_full_training_pipeline(self) -> str:
        """Run the complete training pipeline."""
        print("🎯 Starting full training pipeline...")
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        # Train model
        self.train(train_dataset, val_dataset)
        
        # Evaluate model
        eval_results = self.evaluate(test_dataset)
        
        # Save model
        model_path = self.save_model()
        
        print("🎉 Full training pipeline completed!")
        print(f"📁 Model saved to: {model_path}")
        
        return model_path


def main():
    """Main function to run training."""
    trainer = BotDetectionTrainer()
    model_path = trainer.run_full_training_pipeline()
    print(f"✅ Training completed! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
