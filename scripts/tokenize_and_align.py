import sys
import os
from typing import List, Dict

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def tokenize_and_align_labels(sentences: List[List[str]], 
                               labels: List[List[str]], 
                               tokenizer, 
                               label2id: Dict[str, int]):
    """
    Tokenizes sentences and aligns their POS labels with the generated tokens.
    
    When a word is split into multiple subword tokens (e.g., 'playing' -> 'play', '##ing'),
    the actual label is assigned to the first subword, and -100 is assigned to subsequent 
    subwords and special tokens ([CLS], [SEP]).
    
    Args:
        sentences: List of tokenized sentences.
        labels: List of lists of POS tags.
        tokenizer: HuggingFace Fast Tokenizer.
        label2id: Mapping from tag string to integer ID.
        
    Returns:
        tokenized_inputs: Batched encodings containing input_ids, attention_mask, and labels.
    """
    
    # Tokenize input sentences (split_into_words should be True)
    tokenized_inputs = tokenizer(
        sentences, 
        is_split_into_words=True, 
        truncation=True, 
        padding=True
    )
    
    aligned_labels = []
    
    for i, labels_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word - assign the label
                label_ids.append(label2id[labels_list[word_idx]])
            else:
                # Subsequent subwords - assign -100
                label_ids.append(-100)
            
            previous_word_idx = word_idx
            
        aligned_labels.append(label_ids)
    
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def main():
    """Simple test for tokenization and alignment."""
    from transformers import AutoTokenizer
    
    model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Test data
    sample_sentence = ["हम", "ठीक", "बानी", "।"]
    sample_label = ["PRON", "ADJ", "AUX", "PUNCT"]
    l2i = {"PRON": 0, "ADJ": 1, "AUX": 2, "PUNCT": 3}
    
    print(f"--- Testing Tokenization & Alignment with {model_checkpoint} ---")
    
    try:
        results = tokenize_and_align_labels([sample_sentence], [sample_label], tokenizer, l2i)
        
        tokens = tokenizer.convert_ids_to_tokens(results["input_ids"][0])
        mapped_labels = results["labels"][0]
        
        print(f"Original: {sample_sentence}")
        print(f"Tokens  : {tokens}")
        print(f"Labels  : {mapped_labels}")
        
    except Exception as e:
        print(f"Error during alignment: {e}")

if __name__ == "__main__":
    main()
