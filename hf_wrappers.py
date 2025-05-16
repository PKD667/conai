import torch
import numpy as np
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from sampler.py
from sampler import simplified_sampler, lambda_grammar_parse_fn, passthrough_parse_fn
# Assuming sampler.py is in the same directory, making it a relative import.

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

    def __call__(self, text_list: List[str], return_tensors="pt") -> Dict[str, torch.Tensor]:
        # Assuming simplified_sampler sends one prompt at a time.
        # For hf_tokenizer, "pt" means PyTorch tensor.
        if return_tensors != "pt":
            # This wrapper is now designed to output PyTorch tensors directly.
            raise ValueError("TransformersTokenizerWrapper currently only supports return_tensors='pt' (for PyTorch Tensor output)")

        text = text_list[0]
        
        input_ids_torch: torch.Tensor
        if not text: # Handle empty prompt specifically
            start_token_id = self.hf_tokenizer.bos_token_id
            if start_token_id is None:
                start_token_id = self.hf_tokenizer.eos_token_id
            
            if start_token_id is None:
                raise ValueError("Tokenizer has no BOS or EOS token to start generation from an empty prompt.")
            
            # Shape: (batch_size=1, sequence_length=1), on CPU by default for torch.tensor
            input_ids_torch = torch.tensor([[start_token_id]], dtype=torch.long)
        else:
            # Tokenize non-empty text
            # HF tokenizer returns torch.Tensor on CPU by default.
            encoded = self.hf_tokenizer(text, return_tensors="pt", padding=False, truncation=True)
            input_ids_torch = encoded.input_ids # This is already a torch.Tensor
            
            if input_ids_torch.ndim == 1:
                input_ids_torch = input_ids_torch.unsqueeze(0) 
            
            if input_ids_torch.shape[1] == 0:
                 raise ValueError(f"Tokenization of non-empty prompt '{text}' resulted in zero tokens.")
        
        # Ensure dtype is torch.int32 as expected by some downstream sampler logic (like vocab_indices creation)
        # However, HF models typically expect torch.long (int64).
        # The model wrapper will handle casting to torch.long if needed.
        # For consistency within the sampler's own logic (like full_input_ids), let's use long.
        return {"input_ids": input_ids_torch.to(torch.long)}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text into token IDs."""
        # This method is added to be compatible with parse_fn expecting tokenizer.encode
        return self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, list_of_token_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        return self.hf_tokenizer.batch_decode(list_of_token_ids, skip_special_tokens=skip_special_tokens)

class TransformersModelWrapper:
    def __init__(self, hf_model: AutoModelForCausalLM, device: str):
        self.hf_model = hf_model
        self.device = torch.device(device) # Store as torch.device
        self.hf_model.to(self.device)
        self.hf_model.eval() # Ensure model is in eval mode

    def __call__(self, input_ids: torch.Tensor, past_key_values=None, use_cache=None, attention_mask=None) -> Any:
        # input_ids is now expected to be a torch.Tensor.
        # Ensure it's on the correct device and has the correct dtype (long).
        pt_input_ids = input_ids.to(device=self.device, dtype=torch.long)
        
        pt_attention_mask = None
        if attention_mask is not None:
            # Assuming attention_mask would also be torch.Tensor if provided
            pt_attention_mask = attention_mask.to(device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=pt_input_ids, 
                attention_mask=pt_attention_mask,
                past_key_values=past_key_values, # Assumed to be torch tensors on self.device
                use_cache=use_cache
            )
        
        # outputs.logits is already a torch.Tensor on self.device.
        # No conversion needed.
        
        class Output:
            def __init__(self, lgts, pkvs):
                self.logits = lgts # torch.Tensor
                self.past_key_values = pkvs # PyTorch past_key_values from the hf_model
        
        return Output(outputs.logits, outputs.past_key_values)

    def eval(self): # For compatibility with Hugging Face API and existing code
        self.hf_model.eval()

    def infer(
        self,
        prompt_str: str,
        tokenizer: TransformersTokenizerWrapper, # Pass the tokenizer wrapper instance
        max_think_tokens: int = 50,
        max_formal_tokens: int = 250,
        entropy_limit: int = 8,
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
</system>
<user>identity function for any type A</user>
<think>The user wants the identity function. In typed lambda calculus, this is `λx: A. x`. The type of this function will be `A -> A`.</think>
<formal>λx: (A -> A).x
</formal>
"""

        # Phase 1: Generate "think" content
        think_phase_prompt = f"{system_prompt}\n<user>{prompt_str}</user>\n<think>"
        print(f"\n--- Starting Thinking Phase ---")
        print(f"User query: {prompt_str}\nStarting with: <think>", end="", flush=True)

        # simplified_sampler now returns only the newly generated text part
        newly_generated_think_text, think_stop_seq_found = simplified_sampler(
            model=self,
            tokenizer=tokenizer,
            prompt_str=think_phase_prompt, 
            parse_fn=passthrough_parse_fn,
            max_tokens=max_think_tokens,
            stop_sequences=["</think>"],
            entropy_limit=entropy_limit
        )
        
        think_content_segment = newly_generated_think_text
        if think_stop_seq_found:
            # The newly_generated_think_text already includes the stop sequence
            print(f"\nThink phase ended with stop sequence: {think_stop_seq_found}")
        else:
            # If no stop sequence, ensure the tag is closed
            think_content_segment += "</think>"
            print(f"\nThink phase ended (max_think_tokens: {max_think_tokens} reached). Appending </think>.")
        
        text_after_think_phase = think_phase_prompt + think_content_segment

        # Phase 2: Generate "formal" content
        formal_phase_prompt = text_after_think_phase + "\n<formal>"
        print(f"\n--- Starting Formal Phase ---")
        print(f"Starting with: <formal>", end="", flush=True)

        newly_generated_formal_text, formal_stop_seq_found = simplified_sampler(
            model=self,
            tokenizer=tokenizer,
            prompt_str=formal_phase_prompt,
            parse_fn=lambda_grammar_parse_fn,
            max_tokens=max_formal_tokens,
            stop_sequences=["</formal>"],
            entropy_limit=entropy_limit * 0.5
        )

        formal_content_segment = newly_generated_formal_text
        if formal_stop_seq_found:
            # The newly_generated_formal_text already includes the stop sequence
            print(f"\nFormal phase ended with stop sequence: {formal_stop_seq_found}")
        else:
            # If no stop sequence, ensure the tag is closed
            formal_content_segment += "</formal>"
            print(f"\nFormal phase ended (max_formal_tokens or EOS). Appending </formal>.")
        
        final_full_text = formal_phase_prompt + formal_content_segment
        
        print("\n--- Inference Complete ---")
        # Return only the part from <user> onwards
        user_tag_start = final_full_text.rfind("<user>")
        if user_tag_start != -1:
            return final_full_text[user_tag_start:]
        else:
            # Fallback, though <user> should always be in final_full_text if prompt_str is not empty
            return final_full_text

