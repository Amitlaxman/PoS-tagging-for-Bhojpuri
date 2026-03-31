import torch
from transformers import AutoTokenizer

def test_tokenizer():
    """Sanity check for tokenizer and library installation."""
    
    # Model name
    model_name = "bert-base-multilingual-cased"
    
    # Sample Bhojpuri sentence (using Devnagari script)
    text = "रउआ कइसन बानी? हम ठीक बानी।" # "How are you? I am fine."
    
    print(f"--- Sanity Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Loading Tokenizer: {model_name}\n")
    
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize the sentence
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Full encoding (including special tokens [CLS], [SEP])
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        # Display results
        print(f"Original Text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Full Encoded Input IDs (Tensor): {encoded_dict['input_ids']}")
        
        print(f"\nSuccess! Tokenizer is working correctly.")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_tokenizer()
