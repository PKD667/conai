from tinygrad.tensor import Tensor
from tinygrad import Device,dtypes
from typing import Callable, Tuple, List, Dict, Any
import numpy as np

import typed_lambda_parser

# Type alias for the parsing function
# The first argument to ParseFn is now the 'response generated so far'
ParseFn = Callable[[str, Tuple[str, ...], any, Tensor], Tensor]


def passthrough_parse_fn(
    current_response_text: str,
    all_token_strings: Tuple[str, ...],
    llm_tokenizer: any,
    context_ids: Tensor
) -> Tensor:
    """
    A ParseFn that allows all tokens.
    """
    # Allow all tokens by returning a mask of all True.
    # Ensure the mask is on the same device as context_ids if it matters for downstream ops,
    # though for boolean ops, CPU might be fine. For consistency, use context_ids.device.
    return Tensor.ones(llm_tokenizer.vocab_size, dtype=dtypes.bool, device=context_ids.device)


def lambda_grammar_parse_fn(
    current_response_text: str, # This is now only the response part
    all_token_strings: Tuple[str, ...], 
    llm_tokenizer: any,
    context_ids: Tensor # Full context (prompt + response) token_ids, mainly for device
) -> Tensor: # Returns Tinygrad Tensor
    """
    A ParseFn that uses the typed_lambda_parser to constrain token generation
    based on the current response text.
    """
    mask_np = np.zeros(llm_tokenizer.vocab_size, dtype=bool)

    for i, token_candidate_str in enumerate(all_token_strings):
        # test_expr_str is the current response being built + the candidate token string
        test_expr_str = current_response_text + token_candidate_str
        
        try:
            # If the combined string is empty or all whitespace,
            # allow if the candidate itself is whitespace (parser handles actual spaces).
            if not test_expr_str.strip():
                if any(c.isspace() for c in token_candidate_str):
                    mask_np[i] = True
                continue # Skip further parsing for effectively empty/all-whitespace

            lambda_tokens = typed_lambda_parser.tokenize(test_expr_str)
            parser = typed_lambda_parser.Parser(lambda_tokens)
            parser.parse_expression() # Try to parse the whole thing
            mask_np[i] = True
        except SyntaxError as e:
            # Check if it's a valid prefix (error at/near EOF)
            if hasattr(e, '__self__') and isinstance(e.__self__, typed_lambda_parser.Parser):
                local_parser_instance = e.__self__
                if local_parser_instance.pos >= len(local_parser_instance.tokens) - 1:
                    mask_np[i] = True
            elif "Unexpected character" in str(e): 
                pass # Invalid character, mask_np[i] remains False
            else: # Other syntax errors
                # Allow if current_response_text is empty and candidate is a valid start of an expression
                if not current_response_text.strip():
                    try:
                        # Test candidate alone if current response is empty
                        candidate_lambda_tokens = typed_lambda_parser.tokenize(token_candidate_str)
                        if candidate_lambda_tokens:
                             # Heuristic: if it tokenizes and is a known start token type
                            if candidate_lambda_tokens[0].type in ('IDENTIFIER', 'LAMBDA', 'LPAREN'):
                                mask_np[i] = True
                    except SyntaxError:
                        pass # Candidate alone is not tokenizable or syntactically valid start
        except Exception:
            pass # mask_np[i] remains False

    # Always allow EOS token if it exists and is within vocab size
    if llm_tokenizer.eos_token_id is not None and llm_tokenizer.eos_token_id < llm_tokenizer.vocab_size:
        mask_np[llm_tokenizer.eos_token_id] = True
    
    mask = Tensor(mask_np, dtype=dtypes.bool, device=context_ids.device)

    num_allowed_by_grammar = mask.sum().item()

    # Fallback if grammar is too restrictive and allows no tokens.
    if num_allowed_by_grammar == 0:
        print(f"\nWarning: No tokens allowed by lambda grammar for response prefix: '{current_response_text}'. Allowing all to proceed (DEBUG).")
        mask = Tensor.ones_like(mask, dtype=dtypes.bool) # Fallback to allow all

    return mask


def apply_mask(logits: Tensor, mask: Tensor) -> Tensor:
    """
    Apply a boolean mask to the logits. True in mask means keep, False means discard.
    Logits: (batch_size, vocab_size)
    Mask: (vocab_size,) or (1, vocab_size)
    """
    vocab_size = logits.shape[-1]
    
    # Ensure mask is on the same device and has the correct shape for broadcasting
    # Tinygrad tensors are usually created on Device.DEFAULT
    # Mask should be boolean.
    
    if len(mask.shape) == 1:
        mask_broadcastable = mask.unsqueeze(0) # (1, vocab_size)
    else:
        mask_broadcastable = mask

    # Pad or truncate mask if necessary
    if mask_broadcastable.shape[-1] < vocab_size:
        padding = Tensor.zeros(mask_broadcastable.shape[0], vocab_size - mask_broadcastable.shape[-1], dtype=dtypes.bool, device=logits.device)
        mask_broadcastable = Tensor.cat(mask_broadcastable, padding, dim=1)
    elif mask_broadcastable.shape[-1] > vocab_size:
        mask_broadcastable = mask_broadcastable[:, :vocab_size]

    # `Tensor.where(condition, x, y)`: if condition is true, x, else y.
    # We want to fill with -inf where ~mask is true (i.e., where original mask is false).
    # So, condition is `mask_broadcastable`. Where true, keep logits. Where false, put -inf.
    return Tensor.where(mask_broadcastable, logits, Tensor.full_like(logits, -float("inf")))


def true_next_token_mapping(token_id_tensor: Tensor, tokenizer, prefix_ids: Tensor) -> str:
    """
    Simplified for placeholder: just decodes the token itself.
    token_id_tensor: Tensor of shape (1, 1) containing the token ID.
    """
    # For the placeholder, context doesn't change token string, so decode directly.
    return tokenizer.decode(token_id_tensor.numpy().flatten().tolist(), skip_special_tokens=False)

def batch_true_next_token_mapping(token_ids_batch: Tensor, tokenizer, prefix_ids: Tensor) -> Tuple[str, ...]:
    """
    Simplified for placeholder: decodes each token independently.
    token_ids_batch: Tensor of shape (vocab_size, 1) containing token IDs.
    """
    # For the placeholder, context doesn't change token string.
    decoded_tokens = []
    token_ids_list = token_ids_batch.numpy().astype(int).flatten().tolist()
    for token_id in token_ids_list:
        decoded_tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))
    return tuple(decoded_tokens)


def simplified_sampler(
    model: any,
    tokenizer: any,
    prompt_str: str,
    parse_fn: ParseFn,
    max_tokens: int = 50,
    stop_sequences: List[str] = None,
    # top_k and min_p are not used in this argmax-based sampler
) -> Tuple[str, str | None]: # Return full text up to stop_seq, and the stop_seq found
    # Assuming model wrapper has a 'device' attribute for the Hugging Face model's device
    model_device = getattr(model, 'device', 'N/A')
    print(f"Using device for model inference: {model_device}")
    print(f"Using device for tinygrad tensors in sampler: {Device.DEFAULT}")

    model.eval()

    tokenized_prompt = tokenizer([prompt_str], return_tensors="pt")
    full_input_ids = tokenized_prompt["input_ids"] # Shape: (1, seq_len) on Device.DEFAULT

    # Ensure full_input_ids is 2D [batch_size, seq_len]
    if len(full_input_ids.shape) == 1: # Should be (seq_len,)
        full_input_ids = full_input_ids.unsqueeze(0) # Make it (1, seq_len)
    
    # Handle case where prompt tokenizes to empty but prompt string was not empty.
    # The tokenizer wrapper should ideally prevent this or return BOS.
    if full_input_ids.shape[1] == 0 and prompt_str:
        print(f"Warning: Prompt '{prompt_str}' tokenized to an empty sequence by the wrapper. This may cause issues.")
        # If the model cannot handle empty input_ids (even if batch_size > 0), this will fail.
        # The TransformersModelWrapper __call__ might need to handle input_ids.shape[1] == 0 if tokenizer allows it.
        # However, our tokenizer wrapper now ensures a BOS/EOS token for empty string prompts.
        # This case is for non-empty prompts that tokenize to nothing.

    current_response_str = ""
    found_stop_sequence = None
    
    # Model input is the full sequence (prompt + response so far) as KV caching is not used.
    # The model wrapper handles moving this to the model's device.
    current_model_input_ids = full_input_ids

    for i in range(max_tokens):
        # Model forward pass
        outputs = model(input_ids=current_model_input_ids) # past_key_values=None, use_cache=None implicitly
        
        # Logits for the last token: (batch_size, vocab_size) on Device.DEFAULT
        logits_for_next_token = outputs.logits[:, -1, :] 

        # Prepare all possible next token strings for the parser
        vocab_indices_np = np.arange(tokenizer.vocab_size).reshape(-1, 1)
        # vocab_indices is on Device.DEFAULT
        vocab_indices = Tensor(vocab_indices_np, dtype=dtypes.int32, device=Device.DEFAULT)
        
        # all_token_strings are Python strings. full_input_ids is on Device.DEFAULT.
        all_token_strings = batch_true_next_token_mapping(vocab_indices, tokenizer, full_input_ids)
        
        # Call parse_fn with current_response_str (response part only).
        # custom_mask will be on Device.DEFAULT (from context_ids.device).
        custom_mask = parse_fn(current_response_str, all_token_strings, tokenizer, full_input_ids)
        
        # Apply grammar mask. Logits and mask are on Device.DEFAULT.
        masked_logits = apply_mask(logits_for_next_token, custom_mask)

        if (masked_logits == -float("inf")).all().item() > 0:
            print(f"\nWarning: All tokens masked out by parse_fn for response prefix '{current_response_str}'. Stopping generation.")
            break

        # Greedy sampling (argmax). Result on Device.DEFAULT.
        next_token_index = masked_logits.argmax(axis=-1, keepdim=True) # Shape: (1, 1)

        if tokenizer.eos_token_id is not None and next_token_index.item() == tokenizer.eos_token_id:
            print("\nEOS token generated. Stopping.")
            break
        
        next_token_tensor = next_token_index.cast(dtypes.int32) # Ensure correct dtype, on Device.DEFAULT

        token_text = true_next_token_mapping(next_token_tensor, tokenizer, full_input_ids)
        
        # Add current token to response string *before* checking for stop sequence
        current_response_str += token_text
        
        if stop_sequences:
            for seq in stop_sequences:
                if current_response_str.endswith(seq) or current_response_str[:-1].endswith(seq):
                    found_stop_sequence = seq
                    print(f"\nStop sequence '{seq}' detected.", flush=True)
                    # The loop will break, so the general print below is skipped for this token.
                    break 
            if found_stop_sequence:
                break # Break from generation loop (for i in range(max_tokens))

        # If no stop sequence was found by this token (and loop didn't break), print the token
        print(token_text, end="", flush=True)

        # Update full_input_ids for the next iteration. Concatenation on Device.DEFAULT.
        full_input_ids = Tensor.cat(full_input_ids, next_token_tensor, dim=1)
        current_model_input_ids = full_input_ids # Model gets full sequence
        
        if found_stop_sequence: # Ensure outer loop breaks if stop sequence was found mid-token processing
            break

    return prompt_str + current_response_str, found_stop_sequence

