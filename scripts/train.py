import sys
import os
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from config import config

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_pos_model(model_checkpoint: str, label2id: dict, id2label: dict):
    """
    Loads a pretrained transformer model for token classification.
    
    Args:
        model_checkpoint: The HuggingFace model identifier (e.g., 'bert-base-multilingual-cased').
        label2id: Mapping from tag string to its unique ID.
        id2label: Mapping from unique ID back to its tag string.
        
    Returns:
        model: An initialized AutoModelForTokenClassification.
    """
    
    num_labels = len(label2id)
    
    print(f"Loading Model: {model_checkpoint}")
    print(f"Configuring for {num_labels} unique POS tags.")
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    return model

def get_training_args():
    """Configures the training parameters for fine-tuning."""
    
    return TrainingArguments(
        output_dir=os.path.join(config.MODEL_PATH, "checkpoints"),
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # Recommended practice when using eval_strategy
        report_to="none"             # Avoids unintended integration login prompts
    )

def get_compute_metrics(id2label):
    """Returns a compute_metrics function for the Trainer."""
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index -100 and convert to label strings
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    
    return compute_metrics

def main():
    """Sample demonstrating how to initialize the model using project config."""
    # Example tag set for demonstration
    sample_tags = ["NOUN", "VERB", "PRON", "ADJ", "PUNCT", "AUX"]
    label2id = {tag: i for i, tag in enumerate(sample_tags)}
    id2label = {i: tag for i, tag in enumerate(sample_tags)}
    
    print("--- Initializing Token Classification Model ---")
    
    try:
        # Load using either provided checkpoint or the one from config.py
        model = load_pos_model(
            model_checkpoint=config.MODEL_NAME, 
            label2id=label2id, 
            id2label=id2label
        )
        
        print("\nSuccess! Model loaded and configured.")
        print(f"Device: {model.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
