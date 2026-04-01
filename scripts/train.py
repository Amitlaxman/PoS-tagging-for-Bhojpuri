import sys
import os
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config
from scripts.load_ud_data import load_conllu
from scripts.tokenize_and_align import tokenize_and_align_labels
from utils.label_map import create_label_map

class POSDataset(torch.utils.data.Dataset):
    """Clean PyTorch Dataset for POS Tagging encodings."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def load_pos_model(model_checkpoint: str, label2id: dict, id2label: dict):
    """Loads a pretrained transformer model for token classification."""
    num_labels = len(label2id)
    print(f"Loading Model: {model_checkpoint} with {num_labels} labels.")
    
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
        load_best_model_at_end=True,
        report_to="none"
    )

def get_compute_metrics(id2label):
    """Returns a compute_metrics function for the Trainer."""
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (l) in label if l != -100]
            for label in labels
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    return compute_metrics

def train():
    """Main training pipeline."""
    # 1. Load Data
    print("--- Loading Datasets ---")
    train_file = os.path.join(config.DATA_PATH, "hi_hdtb-ud-train.conllu")
    eval_file = os.path.join(config.DATA_PATH, "hi_hdtb-ud-dev.conllu")
    
    train_sentence, train_labels = load_conllu(train_file)
    eval_sentence, eval_labels = load_conllu(eval_file)
    
    # 2. Create Label Mappings
    label2id, id2label = create_label_map(train_labels)
    
    # 3. Tokenize and Align
    print("--- Tokenizing and Aligning labels ---")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_encodings = tokenize_and_align_labels(train_sentence, train_labels, tokenizer, label2id)
    eval_encodings = tokenize_and_align_labels(eval_sentence, eval_labels, tokenizer, label2id)
    
    # 4. Create PyTorch Datasets
    train_dataset = POSDataset(train_encodings)
    eval_dataset = POSDataset(eval_encodings)
    
    # 5. Load Model
    model = load_pos_model(config.MODEL_NAME, label2id, id2label)
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=get_compute_metrics(id2label)
    )
    
    # 7. Start Training
    print("--- Starting Fine-tuning ---")
    trainer.train()
    
    # 8. Save the final model
    trainer.save_model(os.path.join(config.MODEL_PATH, "final_pos_model"))
    print(f"Model saved to {config.MODEL_PATH}/final_pos_model")

if __name__ == "__main__":
    train()
