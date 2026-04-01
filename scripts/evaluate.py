import sys
import os
from typing import Dict, Any
from transformers import Trainer, PreTrainedModel
from torch.utils.data import Dataset

# Ensure root directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_pos_model(model: PreTrainedModel, dataset: Dataset, trainer: Trainer) -> Dict[str, float]:
    """
    Evaluates a trained POS tagging model on a new dataset using HuggingFace Trainer.

    Args:
        model: The trained POS tagging model.
        dataset: The dataset to evaluate on (already tokenized and aligned).
        trainer: The HuggingFace Trainer instance.

    Returns:
        A dictionary containing precision, recall, f1, and accuracy.
    """
    print(f"--- Running Evaluation on {len(dataset)} samples ---")
    
    # Ensure the trainer is using the provided model and dataset
    # We update the model in the trainer in case it was modified after initialization
    trainer.model = model
    
    # Run evaluation
    metrics = trainer.evaluate(eval_dataset=dataset)
    
    # HuggingFace Trainer prefixes metric keys with 'eval_' 
    # Extract them while handling the naming convention used in train.py
    results = {
        "precision": metrics.get("eval_precision"),
        "recall": metrics.get("eval_recall"),
        "f1": metrics.get("eval_f1"),
        "accuracy": metrics.get("eval_accuracy")
    }

    # Print results for visibility
    print("\nEvaluation Results:")
    for metric, value in results.items():
        if value is not None:
             print(f"  {metric.capitalize()}: {value:.4f}")
        else:
             print(f"  {metric.capitalize()}: Not found in trainer output.")

    return results

if __name__ == "__main__":
    # This section can be expanded to load a model and dataset for a standalone run
    print("Function 'evaluate_pos_model' is ready for use within the pipeline.")
