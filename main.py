# Import the typed lambda calculus parser components
import typed_lambda_parser # Assuming this file exists and is correct

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Import wrappers from the new file
from hf_wrappers import TransformersTokenizerWrapper, TransformersModelWrapper

from sampler import simplified_sampler,lambda_grammar_parse_fn

if __name__ == "__main__":
    # The lambda_vocab was for the PlaceholderTokenizer.
    # The AutoTokenizer uses its own pre-trained vocabulary.

    # --- Use Hugging Face Transformers model and tokenizer ---
    # model_name = "gpt2" # Example: GPT-2
    #model_name = "EleutherAI/gpt-neo-125M" # Example: GPT-Neo
    #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Example: TinyLlama, ensure you have transformers>=4.34 for Llama2 tokenizers
    #model_name = "google/gemma-3-4b-it"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print(f"Loading Hugging Face tokenizer: {model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name,token=os.getenv("HUGGINGFACE_TOKEN"))
    print(f"Loading Hugging Face model: {model_name}")
    # Specify torch_dtype for model loading if desired, e.g., torch.bfloat16 or torch.float16
    # Ensure the chosen dtype is supported by the model and hardware.
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using model dtype: {model_dtype}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        torch_dtype=model_dtype # Use the determined dtype
    )
    
    # Set device (CUDA if available, otherwise CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device_str}")
    # Wrap the Hugging Face components for compatibility with simplified_sampler
    wrapped_tokenizer = TransformersTokenizerWrapper(hf_tokenizer)
    # Pass the device string; TransformersModelWrapper will create torch.device(device_str)
    wrapped_model = TransformersModelWrapper(hf_model, device_str)

    # do some testing
    # Example input
    input_text = "Write the succesor function in typed lambda calculus."
    # inference
    # Assuming the model is a causal language model
    # Use the model to generate predictions
    # Note: The model should be in evaluation mode (set in TransformersModelWrapper)
    # wrapped_model.eval() # Already called in TransformersModelWrapper.__init__

    content = wrapped_model.infer(
        prompt_str=input_text,
        tokenizer=wrapped_tokenizer,
        entropy_limit=4
        ) 
    print(content)