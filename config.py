import torch
import os

class Config:
    """Configuration class for Bhojpuri POS Tagging Project."""
    
    # Model Configuration
    MODEL_NAME = "bert-base-multilingual-cased"
    MAX_LEN = 128           # Adjust based on your Bhojpuri sentence complexity
    TRAIN_BATCH_SIZE = 16   # Optimized for most GPUs
    VALID_BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    EPOCHS = 5
    
    # Device Auto-detection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Folder Structure
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data")
    MODEL_PATH = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Label Configuration (to be populated during preprocessing)
    LABELS = []  # e.g., ["NOUN", "VERB", "ADJ", ...]
    NUM_LABELS = 0

    # Ensure output directories exist
    @staticmethod
    def create_dirs():
        for path in [Config.DATA_PATH, Config.MODEL_PATH, Config.OUTPUT_DIR]:
            if not os.path.exists(path):
                os.makedirs(path)

# Initialize global config instance
config = Config()

if __name__ == "__main__":
    print(f"--- Project Configuration ---")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Training on: {config.EPOCHS} epochs")
