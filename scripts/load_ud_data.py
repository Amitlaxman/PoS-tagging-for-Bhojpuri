import os
from typing import List, Tuple

def load_conllu(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Robustly loads a Universal Dependencies (UD) dataset in CoNLL-U format.
    
    Args:
        file_path: Path to the .conllu file.
        
    Returns:
        A tuple containing (sentences, labels) where:
        - sentences: List of lists of tokens (FORM)
        - labels: List of lists of POS tags (UPOS)
    """
    sentences = []
    labels = []
    
    curr_tokens = []
    curr_tags = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Handle sentence boundaries (empty lines)
            if not line:
                if curr_tokens:
                    sentences.append(curr_tokens)
                    labels.append(curr_tags)
                    curr_tokens = []
                    curr_tags = []
                continue
            
            # UD use tab-separated columns
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            
            idx = parts[0]
            form = parts[1]
            upos = parts[3]
            
            # Rule 1: Ignore rows where ID contains "-" (Multi-word tokens)
            # Rule 2: Ignore rows where FORM is "_" (Under-specified words)
            if '-' in idx or form == "_":
                continue
            
            curr_tokens.append(form)
            curr_tags.append(upos)
            
        # Final append for files missing a terminal newline
        if curr_tokens:
            sentences.append(curr_tokens)
            labels.append(curr_tags)
            
    return sentences, labels

def main():
    """Test functionality on the sample dataset."""
    sample_path = os.path.join("data", "bho_bhtb-ud-test.conllu")
    
    if not os.path.exists(sample_path):
        print(f"Warning: Sample file {sample_path} not found. Creating a quick mock for test.")
        # Minimal mock content
        mock_content = (
            "1\tरउआ\t_\tPRON\t_\t_\t_\t_\t_\t_\n"
            "2\tकइसन\t_\tADJ\t_\t_\t_\t_\t_\t_\n"
            "3\tबानी\t_\tAUX\t_\t_\t_\t_\t_\t_\n"
            "4\t?\t_\tPUNCT\t_\t_\t_\t_\t_\t_\n\n"
        )
        os.makedirs("data", exist_ok=True)
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(mock_content)

    print(f"--- Loading UD CoNLL-U formatted data ---")
    try:
        sentences, labels = load_conllu(sample_path)
        print(f"Dataset Size: {len(sentences)} sentences found.\n")
        
        # Display first 2 sentences and labels for verification
        for i in range(min(2, len(sentences))):
            print(f"Sentence {i+1}: {' '.join(sentences[i])}")
            print(f"Labels   {i+1}: {labels[i]}\n")
            
    except Exception as e:
        print(f"Error during loading: {e}")

if __name__ == "__main__":
    main()
