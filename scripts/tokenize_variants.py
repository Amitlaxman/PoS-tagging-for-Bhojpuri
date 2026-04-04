def tokenize_and_align_labels_variant(sentences, labels, tokenizer, label2id, strategy="first"):
    """
    Tokenizes sentences and aligns POS labels with different alignment strategies.
    
    Args:
        sentences (list of list of strings): Sentences as list of words.
        labels (list of list of strings): Labels corresponding to words.
        tokenizer: HuggingFace tokenizer.
        label2id (dict): Mapping from label string to integer ID.
        strategy (str): Alignment strategy ("first", "all", "last").
        
    Returns:
        dict: encodings with input_ids, attention_mask, and labels
    """
    # Tokenize the input sentences
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True
    )
    
    aligned_labels = []
    
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        
        # Pre-compute last subword indices if strategy is 'last'
        last_subword_indices = set()
        if strategy == "last":
            for idx in range(len(word_ids) - 1):
                if word_ids[idx] is not None and word_ids[idx] != word_ids[idx + 1]:
                    last_subword_indices.add(idx)
            # Always check the last token to see if it's the end of a word
            if len(word_ids) > 0 and word_ids[-1] is not None:
                last_subword_indices.add(len(word_ids) - 1)
        
        label_ids = []
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            # Assign -100 to special tokens so they are ignored in the loss function
            if word_idx is None:
                label_ids.append(-100)
            else:
                if strategy == "first":
                    # Label only the first subword of a word, rest = -100
                    if word_idx != previous_word_idx:
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                        
                elif strategy == "all":
                    # Assign the same label to all subwords of a word
                    label_ids.append(label2id[label[word_idx]])
                    
                elif strategy == "last":
                    # Label only the last subword of a word, rest = -100
                    # Check if next token is part of the same word
                    is_last = False
                    if idx + 1 < len(word_ids):
                        if word_ids[idx + 1] != word_idx:
                            is_last = True
                    else:
                        is_last = True # It's the last token in the sequence
                    
                    if is_last:
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                        
                else:
                    raise ValueError(f"Unknown strategy: {strategy}. Use 'first', 'all', or 'last'.")
                    
            previous_word_idx = word_idx
            
        aligned_labels.append(label_ids)
        
    # Add the aligned labels to the encodings dictionary
    encodings["labels"] = aligned_labels
    return encodings


def debug_alignment(sentences, labels, tokenizer, label2id, strategy="first"):
    """
    Debug function to print original tokens, subwords, and aligned labels.
    """
    id2label = {v: k for k, v in label2id.items()}
    encodings = tokenize_and_align_labels_variant(sentences, labels, tokenizer, label2id, strategy)
    
    print(f"\n--- Strategy: {strategy} ---")
    for i, (sentence, label_list) in enumerate(zip(sentences, labels)):
        print(f"\nExample {i+1}:")
        print(f"Original: {' '.join(sentence)}")
        
        tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][i])
        aligned_label_ids = encodings["labels"][i]
        
        # Determine column widths
        token_width = max(len(t) for t in tokens) + 2
        label_width = max(len(str(id2label.get(lid, lid))) for lid in aligned_label_ids) + 2
        
        header = f"{'Token':<{token_width}} | {'Label ID':<10} | {'Label Name':<{label_width}}"
        print(header)
        print("-" * len(header))
        
        for token, lid in zip(tokens, aligned_label_ids):
            label_name = id2label.get(lid, "-100") if lid != -100 else "-100"
            print(f"{token:<{token_width}} | {lid:<10} | {label_name:<{label_width}}")
