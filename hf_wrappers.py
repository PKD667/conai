from tinygrad.tensor import Tensor
from tinygrad import dtypes
import numpy as np
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import from sampler.py
from sampler import simplified_sampler, lambda_grammar_parse_fn, passthrough_parse_fn
# Assuming sampler.py is in the same directory, making it a relative import.
# If not, adjust the import path accordingly (e.g., `import sampler` and then `sampler.simplified_sampler`).

# --- Wrapper classes for Hugging Face Transformers ---

class TransformersTokenizerWrapper:
    def __init__(self, hf_tokenizer: AutoTokenizer):
        self.hf_tokenizer = hf_tokenizer
        
        # Ensure pad_token is set. For many causal LMs, it's common to use eos_token as pad_token.
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id

        self.pad_token = self.hf_tokenizer.pad_token
        self.eos_token = self.hf_tokenizer.eos_token
        self.unk_token = self.hf_tokenizer.unk_token
            
        self.pad_token_id = self.hf_tokenizer.pad_token_id
        self.eos_token_id = self.hf_tokenizer.eos_token_id
        # Some tokenizers might not have a specific unk_token_id, handle if necessary
        self.unk_token_id = self.hf_tokenizer.unk_token_id if self.hf_tokenizer.unk_token_id is not None else self.hf_tokenizer.eos_token_id
        
        self.vocab_size = self.hf_tokenizer.vocab_size

    def __call__(self, text_list: List[str], return_tensors="pt") -> Dict[str, Tensor]:
        # Assuming simplified_sampler sends one prompt at a time.
        # For hf_tokenizer, "pt" means PyTorch tensor.
        if return_tensors != "pt":
            raise ValueError("TransformersTokenizerWrapper currently only supports return_tensors='pt' (for tinygrad Tensor output)")

        text = text_list[0]
        
        input_ids_np: np.ndarray
        if not text: # Handle empty prompt specifically
            # Use BOS token if available, otherwise EOS as a fallback start token for generation.
            start_token_id = self.hf_tokenizer.bos_token_id
            if start_token_id is None:
                start_token_id = self.hf_tokenizer.eos_token_id
            
            if start_token_id is None:
                # This case should be rare for generative models.
                # If truly no BOS/EOS, model might not support unconditional generation well.
                # As a last resort, could use a common token like space, but BOS/EOS is preferred.
                raise ValueError("Tokenizer has no BOS or EOS token to start generation from an empty prompt.")
            
            # Shape: (batch_size=1, sequence_length=1)
            input_ids_np = np.array([[start_token_id]], dtype=np.int32)
        else:
            # Tokenize non-empty text
            encoded = self.hf_tokenizer(text, return_tensors="pt", padding=False, truncation=True)
            input_ids_torch = encoded.input_ids
            
            # Ensure input_ids is 2D: [batch_size, sequence_length]
            # hf_tokenizer with a single string and return_tensors="pt" usually returns 2D.
            if input_ids_torch.ndim == 1:
                input_ids_torch = input_ids_torch.unsqueeze(0) # Make it (1, seq_len)
            
            # Handle cases where tokenization of non-empty string might still result in empty tensor (highly unlikely for valid strings)
            if input_ids_torch.shape[1] == 0:
                 raise ValueError(f"Tokenization of non-empty prompt '{text}' resulted in zero tokens.")

            input_ids_np = input_ids_torch.cpu().numpy().astype(np.int32)
        
        # Convert NumPy array to tinygrad Tensor
        # tinygrad.Tensor by default is on Device.DEFAULT (usually CPU)
        return {"input_ids": Tensor(input_ids_np, dtype=dtypes.int32)}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, list_of_token_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        return self.hf_tokenizer.batch_decode(list_of_token_ids, skip_special_tokens=skip_special_tokens)

class TransformersModelWrapper:
    def __init__(self, hf_model: AutoModelForCausalLM, device: str):
        self.hf_model = hf_model
        self.device = device
        self.hf_model.to(self.device)
        self.hf_model.eval() # Ensure model is in eval mode

    def __call__(self, input_ids: Tensor, past_key_values=None, use_cache=None, attention_mask=None) -> Any:
        # Convert tinygrad.Tensor to PyTorch tensor
        # input_ids from tokenizer wrapper are int32. Transformers models expect int64 (torch.long).
        input_ids_np = input_ids.numpy() 
        pt_input_ids = torch.tensor(input_ids_np, dtype=torch.long).to(self.device)
        
        pt_attention_mask = None
        if attention_mask is not None:
            # Assuming attention_mask would also be tinygrad.Tensor if provided
            pt_attention_mask = torch.tensor(attention_mask.numpy(), dtype=torch.long).to(self.device)
        
        # past_key_values are expected to be in the format returned by the Hugging Face model
        # (which are PyTorch tensors on the correct device if use_cache=True was used previously)
        
        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=pt_input_ids, 
                attention_mask=pt_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
        
        # Convert PyTorch logits tensor back to tinygrad.Tensor
        # Logits should be float32.
        # Move to CPU before converting to numpy, then to tinygrad.Tensor.
        logits_np = outputs.logits.detach().cpu().numpy().astype(np.float32)
        tg_logits = Tensor(logits_np, dtype=dtypes.float32) # tinygrad infers device (default CPU)
        
        class Output:
            def __init__(self, lgts, pkvs):
                self.logits = lgts
                self.past_key_values = pkvs # These are PyTorch past_key_values from the hf_model
        
        return Output(tg_logits, outputs.past_key_values)

    def eval(self): # For compatibility with Hugging Face API and existing code
        self.hf_model.eval()

    def infer(
        self,
        prompt_str: str,
        tokenizer: TransformersTokenizerWrapper, # Pass the tokenizer wrapper instance
        max_think_tokens: int = 50,
        max_formal_tokens: int = 50
    ) -> str:
        """
        Generates a response in two phases:
        1. Thinking phase: LLM generates content within <think>...</think> tags.
        2. Formal phase: LLM generates content within <formal>...</formal> tags,
                         constrained by lambda_grammar_parse_fn.
        """
        
        system_prompt = """<system>
You are an AI assistant that translates natural language queries into formal typed lambda calculus expressions.
The interaction will follow this structure:
1. The user's query will be provided within <user>...</user> tags.
2. First, you will engage in a "thinking" step. Your thoughts and reasoning should be enclosed in <think>...</think> tags. This is your scratchpad to break down the problem, plan the lambda expression, or note any ambiguities. You MUST explicitly close this tag with </think> when your thinking process is complete.
3. After closing the </think> tag, you MUST open a <formal> tag.
4. Inside the <formal>...</formal> tags, you will write ONLY the typed lambda calculus expression corresponding to the user's query.
5. You MUST explicitly close the formal section with </formal> once the lambda expression is complete.

Example of a complete interaction:
<user>identity function for any type A</user><think>The user wants the identity function. In typed lambda calculus, this is `λx: A. x`. The type of this function will be `A -> A`.</think><formal>λx: (A -> A).x</formal>
</system>"""

        # Phase 1: Generate "think" content
        think_prompt_prefix = f"{system_prompt}\n<user>{prompt_str}</user>\n<think>"
        print(f"\n--- Starting Thinking Phase ---")
        # Print only the user-relevant part of the prompt for brevity in logs
        print(f"User query: {prompt_str}\nStarting with: <think>", end="", flush=True)


        generated_think_text_base, think_stop_seq_found = simplified_sampler(
            model=self,
            tokenizer=tokenizer,
            prompt_str=think_prompt_prefix,
            parse_fn=passthrough_parse_fn, # Allow free generation for thoughts
            max_tokens=max_think_tokens,
            stop_sequences=["</think>"]
        )
        
        current_full_text = generated_think_text_base
        if think_stop_seq_found:
            current_full_text += think_stop_seq_found 
            print(f"\nThink phase ended with stop sequence: {think_stop_seq_found}")
        else:
            current_full_text += "</think>" 
            print(f"\nThink phase ended (max_think_tokens: {max_think_tokens} reached). Appending </think>.")
        
        # Phase 2: Generate "formal" content
        formal_prompt_prefix = current_full_text + "\n<formal>"
        print(f"\n--- Starting Formal Phase ---")
        print(f"Starting with: <formal>", end="", flush=True)


        generated_formal_text_base, formal_stop_seq_found = simplified_sampler(
            model=self,
            tokenizer=tokenizer,
            prompt_str=formal_prompt_prefix,
            parse_fn=lambda_grammar_parse_fn, # Constrain with lambda grammar
            max_tokens=max_formal_tokens,
            stop_sequences=["</formal>"] # Enable detection of </formal>
        )

        final_text = generated_formal_text_base
        if formal_stop_seq_found:
            final_text += formal_stop_seq_found 
            print(f"\nFormal phase ended with stop sequence: {formal_stop_seq_found}")
        else:
            final_text += "</formal>" 
            print(f"\nFormal phase ended (max_formal_tokens or EOS). Appending </formal>.")
        
        print("\n--- Inference Complete ---")
        # Return only the part from <user> onwards, excluding the system prompt for cleaner final output
        # Find the start of the <user> tag in the final_text
        user_tag_start = final_text.find("<user>")
        if user_tag_start != -1:
            return final_text[user_tag_start:]
        else:
            # Should not happen if prompt_str is non-empty and structure is followed
            return final_text

