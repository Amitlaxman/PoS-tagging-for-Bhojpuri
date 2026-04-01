import random
from typing import List, Tuple

def train_test_split_data(
    sentences: List[List[str]], 
    labels: List[List[str]], 
    split_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Shuffles and splits a dataset into training and testing sets.
    
    Args:
        sentences: List of tokenized sentences.
        labels: List of lists of POS tags.
        split_ratio: Percentage of data to use for training (default 0.8).
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple of (train_sentences, train_labels, test_sentences, test_labels).
    """
    if len(sentences) != len(labels):
        raise ValueError("Sentences and labels must have the same length.")
    
    # Combine to shuffle together
    combined = list(zip(sentences, labels))
    random.seed(seed)
    random.shuffle(combined)
    
    # Split back
    shuffled_sentences, shuffled_labels = zip(*combined)
    
    split_idx = int(len(shuffled_sentences) * split_ratio)
    
    train_sentences = list(shuffled_sentences[:split_idx])
    train_labels = list(shuffled_labels[:split_idx])
    
    test_sentences = list(shuffled_sentences[split_idx:])
    test_labels = list(shuffled_labels[split_idx:])
    
    print(f"Dataset split completed: {len(train_sentences)} train samples, {len(test_sentences)} test samples.")
    
    return train_sentences, train_labels, test_sentences, test_labels

if __name__ == "__main__":
    # Test the split functionality
    sample_s = [["word1"], ["word2"], ["word3"], ["word4"], ["word5"]]
    sample_l = [["tag1"], ["tag2"], ["tag3"], ["tag4"], ["tag5"]]
    
    tr_s, tr_l, te_s, te_l = split_dataset(sample_s, sample_l, split_ratio=0.6)
    print(f"Train sentences: {len(tr_s)}")
    print(f"Test sentences:  {len(te_s)}")
