import sys
import os
import json
from typing import List

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.load_ud_data import load_conllu
from utils.label_map import create_label_map

def save_as_json(sentences: List[List[str]], labels: List[List[str]], output_path: str):
    """
    Saves tokenized sentences and labels into a structured JSON file.
    
    Format:
    [
      {"tokens": ["token1", "token2"], "labels": ["tag1", "tag2"]},
      ...
    ]
    """
    processed_data = []
    
    for tokens, tags in zip(sentences, labels):
        processed_data.append({
            "tokens": tokens,
            "labels": tags
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(processed_data)} items to {output_path}")

def run_preprocessing():
    """Main preprocessing pipeline for the project."""
    
    # Input file from UD project structure
    input_file = os.path.join("data", "hi_hdtb-ud-train.conllu")
    output_file = os.path.join("data", "processed_hindi.json")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Please ensure your dataset is placed correctly.")
        return
        
    print(f"--- Preprocessing: Converting {input_file} to JSON ---")
    
    try:
        # 1. Load the raw data from CoNLL-U format
        sentences, labels = load_conllu(input_file)
        
        # 2. (Optional) Generate label mappings for debugging/reference
        l2i, i2l = create_label_map(labels)
        print(f"Unique labels in training set: {len(l2i)}")
        
        # 3. Save to refined JSON structure
        save_as_json(sentences, labels, output_file)
        
    except Exception as e:
        print(f"Failed to preprocess data: {e}")

if __name__ == "__main__":
    run_preprocessing()
