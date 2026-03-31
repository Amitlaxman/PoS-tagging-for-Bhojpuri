from typing import List, Dict, Tuple

def create_label_map(labels: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates consolidated label2id and id2label mappings from POS tag sequences.
    
    Args:
        labels: A nested list of tags (List[List[str]]).
        
    Returns:
        A tuple of (label2id, id2label) dictionaries sorted alphabetically 
        to ensure deterministic mapping across runs.
    """
    
    # Flatten the list and extract unique tags
    unique_tags = set()
    for sentence_tags in labels:
        for tag in sentence_tags:
            unique_tags.add(tag)
            
    # Sort for consistent mapping
    sorted_tags = sorted(list(unique_tags))
    
    # Create bidirectional mappings
    label2id = {tag: i for i, tag in enumerate(sorted_tags)}
    id2label = {i: tag for i, tag in enumerate(sorted_tags)}
    
    return label2id, id2label

def main():
    """Simple test for label mapping logic."""
    test_labels = [
        ["NOUN", "VERB", "ADJ", "NOUN"],
        ["PRON", "VERB", "NOUN", "PUNCT"]
    ]
    
    print("--- Testing Label Mapping ---")
    l2i, i2l = create_label_map(test_labels)
    
    print(f"Unique Labels found: {len(l2i)}")
    print(f"label2id: {l2i}")
    print(f"id2label: {i2l}")

if __name__ == "__main__":
    main()
