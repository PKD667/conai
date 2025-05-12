# Import the typed lambda calculus parser components
import typed_lambda_parser # Assuming this file exists and is correct

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import wrappers from the new file
from hf_wrappers import TransformersTokenizerWrapper, TransformersModelWrapper

from sampler import simplified_sampler,lambda_grammar_parse_fn

if __name__ == "__main__":
    # The lambda_vocab was for the PlaceholderTokenizer.
    # The AutoTokenizer uses its own pre-trained vocabulary.

    # --- Use Hugging Face Transformers model and tokenizer ---
    # model_name = "gpt2" # Example: GPT-2
    model_name = "EleutherAI/gpt-neo-125M" # Example: GPT-Neo
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Example: TinyLlama, ensure you have transformers>=4.34 for Llama2 tokenizers

    print(f"Loading Hugging Face tokenizer: {model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading Hugging Face model: {model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set device (CUDA if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    # Wrap the Hugging Face components for compatibility with simplified_sampler
    wrapped_tokenizer = TransformersTokenizerWrapper(hf_tokenizer)
    wrapped_model = TransformersModelWrapper(hf_model, device)

    # do some testing
    # Example input
    input_text = "Write the first church number in lambda calculus."
    # inference
    # Assuming the model is a causal language model
    # Use the model to generate predictions
    # Note: The model should be in evaluation mode
    wrapped_model.eval()

    content = wrapped_model.infer(prompt_str=input_text,tokenizer=wrapped_tokenizer) 
    print(content)