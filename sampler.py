import torch
from typing import Callable, Tuple, List, Dict, Any
import numpy as np

import typed_lambda_parser
from typed_lambda_parser import TypeCheckError, ParserSyntaxError, Token

# Type alias for the parsing function
# The first argument to ParseFn is now the 'response generated so far'
ParseFn = Callable[[str, Tuple[str, ...], any, torch.Tensor], torch.Tensor]


def passthrough_parse_fn(
    current_response_text: str,
    all_token_strings: Tuple[str, ...],
    llm_tokenizer: any,
    context_ids: torch.Tensor
) -> torch.Tensor:
    """
    A ParseFn that allows all tokens.
    """
    # Allow all tokens by returning a mask of all True.
    # Ensure the mask is on the same device as context_ids.
    # Size the mask based on the actual number of token strings provided.
    return torch.ones(len(all_token_strings), dtype=torch.bool, device=context_ids.device)

def lambda_grammar_parse_fn(
    current_response_text: str, 
    all_token_strings: Tuple[str, ...], 
    llm_tokenizer: any,
    context_ids: torch.Tensor
) -> torch.Tensor:
    """
    A ParseFn that uses the typed_lambda_parser to constrain token generation
    based on the current response text, including type checking.
    EOS token is handled by the sampler after this function returns.
    """
    # Create a mask initialized to all False, sized by len(all_token_strings)
    mask_np = np.zeros(len(all_token_strings), dtype=bool)
    
    # Normalize lambda characters in the current response
    normalized_response = current_response_text.replace('λ', 'lambda')
    
    # Process each token candidate
    for i, token_str in enumerate(all_token_strings):
        normalized_token = token_str.replace('λ', 'lambda')
        mask_np[i] = is_valid_token_candidate(normalized_response, normalized_token)
    
    # EOS token handling is moved to the sampler.
    
    # Log warning if no tokens are allowed by the grammar
    log_mask_warnings(mask_np, current_response_text, all_token_strings)
    
    # Convert to torch tensor and return
    return torch.tensor(mask_np, dtype=torch.bool, device=context_ids.device)


def is_valid_token_candidate(current_text: str, token_str: str) -> bool:
    """Determine if a token is valid to append to the current text."""
    
    test_expr_combined = current_text + token_str
    test_expr_stripped = test_expr_combined.strip()

    # 1. If the combined expression is empty or all whitespace,
    #    it's a valid "empty" or "whitespace" prefix.
    #    This allows starting with whitespace or adding more whitespace.
    if not test_expr_stripped:
        return True

    try:
        # Tokenize the combined expression
        lambda_tokens = typed_lambda_parser.tokenize(test_expr_combined)

        # If tokenization results in no tokens (e.g., test_expr was only comments or complex whitespace)
        # and the stripped expression was not empty, this is an invalid state.
        if not lambda_tokens and test_expr_stripped:
            print(f"Tokenization resulted in no tokens for '{test_expr_combined}'. Invalid expression.")
            return False
        
        # If there are no tokens and the string was empty/whitespace, it's caught by the first check.
        # If there are no tokens from a non-empty, non-whitespace string, it's invalid.
        if not lambda_tokens:
            print(f"Tokenization resulted in no tokens for '{test_expr_combined}'. Invalid expression.")
            return False

        parser = typed_lambda_parser.Parser(lambda_tokens)
        ast = parser.parse_expression() # Attempt to parse one expression
        
        # 2. NEW CHECK: Ensure all tokens were consumed by parse_expression.
        # If parser.pos < len(parser.tokens), it means parse_expression() succeeded
        # but did not consume all available tokens. This implies trailing characters
        # that are not part of the *single* parsed expression (e.g., "(x) y").
        if parser.tokens[parser.pos] != Token(type="EOF",value="") and parser.pos < len(parser.tokens):
            print(f"Parser did not consume all tokens for '{test_expr_combined}'. Remaining tokens: {parser.tokens[parser.pos:]}")
            return False

        # 3. All tokens were consumed, now try to type-check the AST.
        try:
            typed_lambda_parser.infer_type(ast, {})
            return True  # Valid expression, all tokens consumed, correct types.
        except TypeCheckError:
            return False # Valid syntax, all tokens consumed, but a type error.
            
    except ParserSyntaxError as e:
        # A ParserSyntaxError means the current test_expr_combined is not a fully valid expression.
        # Check if it's a valid prefix of a potentially valid expression.
        if is_valid_prefix(e):
            # The expression is a valid prefix (error occurred at/near the end).
            # Additional check: if current_text is empty, the token_str itself
            # must be a valid start of an expression.
            # is_valid_prefix alone might be true for single non-starting tokens like ')' or ':'.
            if not current_text.strip() and not could_start_expression(token_str):
                # Example: current_text="", token_str=")". test_expr_combined=")"
                # is_valid_prefix(e) is True for ")", but could_start_expression(")") is False.
                return False
            return True # It's a valid prefix.
        else:
            # The syntax error occurred before the end of test_expr_combined,
            # meaning it's not just incomplete but malformed earlier.
            return False
        
    except Exception: # Catch-all for other unexpected errors during parsing/tokenization.
        return False  # Fail closed on unexpected errors


def is_valid_prefix(parser_error: ParserSyntaxError) -> bool:
    """Check if the parsing error occurred at the end, indicating a potential prefix."""
    if parser_error.parser_instance is None:
        return False
        
    parser_instance = parser_error.parser_instance
    return parser_instance.pos >= len(parser_instance.tokens) - 1


def could_start_expression(token_str: str) -> bool:
    """Check if the token could potentially start a valid lambda expression."""
    try:
        tokens = typed_lambda_parser.tokenize(token_str)
        if not tokens:
            return False
            
        # Valid expression starters
        return tokens[0].type in ('IDENTIFIER', 'LAMBDA', 'LPAREN')
    except:
        return False


def log_mask_warnings(mask_np: np.ndarray, current_response_text: str, all_token_strings: Tuple[str, ...]) -> None:
    """Log warnings if the mask is too restrictive. Assumes EOS is handled by sampler."""
    num_allowed = np.sum(mask_np)
    
    if num_allowed == 0:
        print(f"\nWarning: No tokens allowed by lambda grammar/type rules for: '{current_response_text}'. Generation will likely stop or rely on EOS if sampler allows it.")
    elif num_allowed == 1:
        allowed_idx = -1
        # Find the single allowed token string if possible (mask_np might be all False if num_allowed is 0 due to float precision)
        if np.any(mask_np):
            allowed_idx = np.where(mask_np)[0][0]
            allowed_token_str = all_token_strings[allowed_idx]
            print(f"\nWarning: Only 1 token ('{allowed_token_str}') allowed by lambda grammar for: '{current_response_text}'.")
        else: # Should not happen if num_allowed == 1, but as a safeguard
            print(f"\nWarning: Only 1 token allowed by lambda grammar (but couldn't identify which) for: '{current_response_text}'.")

    elif num_allowed < 5:  # Arbitrary small number to warn about very restricted choices
        print(f"\nWarning: Very few tokens ({num_allowed}) allowed for: '{current_response_text}'")
        # print the allowed tokens
        allowed_indices = np.where(mask_np)[0]
        allowed_tokens = [all_token_strings[i] for i in allowed_indices]
        print(f"Allowed token strings: {allowed_tokens}")
        # For debugging, you might also want to see their indices:
        # print(f"Allowed token indices: {allowed_indices.tolist()}")


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a boolean mask to the logits. True in mask means keep, False means discard.
    Logits: (batch_size, vocab_size)
    Mask: (vocab_size,) or (1, vocab_size)
    """
    vocab_size = logits.shape[-1]
    
    mask = mask.to(logits.device)
    
    if len(mask.shape) == 1:
        mask_broadcastable = mask.unsqueeze(0)
    else:
        mask_broadcastable = mask

    if mask_broadcastable.shape[-1] < vocab_size:
        padding_shape = list(mask_broadcastable.shape)
        padding_shape[-1] = vocab_size - mask_broadcastable.shape[-1]
        padding = torch.zeros(padding_shape, dtype=torch.bool, device=logits.device)
        mask_broadcastable = torch.cat((mask_broadcastable, padding), dim=-1)
    elif mask_broadcastable.shape[-1] > vocab_size:
        mask_broadcastable = mask_broadcastable[..., :vocab_size]

    return torch.where(mask_broadcastable, logits, torch.full_like(logits, -float("inf")))


def true_next_token_mapping(token_id_tensor: torch.Tensor, tokenizer, prefix_ids: torch.Tensor) -> str:
    """
    Simplified for placeholder: just decodes the token itself.
    token_id_tensor: Tensor of shape (1, 1) containing the token ID.
    """
    return tokenizer.decode(token_id_tensor.cpu().numpy().flatten().tolist(), skip_special_tokens=False)

def batch_true_next_token_mapping(token_ids_batch: torch.Tensor, tokenizer, prefix_ids: torch.Tensor) -> Tuple[str, ...]:
    """
    Simplified for placeholder: decodes each token independently.
    token_ids_batch: Tensor of shape (vocab_size, 1) containing token IDs.
    """
    decoded_tokens = []
    token_ids_list = token_ids_batch.cpu().numpy().astype(int).flatten().tolist()
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
    entropy_limit: float = 0.5,
    temperature: float = 1.0,
    top_k: int = None,
    min_p: float = 0.0
) -> Tuple[str, str | None]:
    # Determine model's device more robustly
    if hasattr(model, 'device') and isinstance(model.device, torch.device):
        model_device = model.device
    else:
        # Attempt to get device string and convert, with fallback
        model_device_str = str(getattr(model, 'device', 'cpu')) 
        try:
            model_device = torch.device(model_device_str)
        except RuntimeError:
            print(f"Warning: Could not determine model device from '{model_device_str}'. Defaulting to CPU.")
            model_device = torch.device('cpu')
    
    print(f"Using device for model inference: {model_device}")

    model.eval()

    # Tokenize prompt and ensure it's on CPU for consistent concatenation later.
    tokenized_prompt = tokenizer([prompt_str], return_tensors="pt") 
    full_input_ids = tokenized_prompt["input_ids"].to('cpu') # Keep accumulated IDs on CPU

    if len(full_input_ids.shape) == 1:
        full_input_ids = full_input_ids.unsqueeze(0)
    
    if full_input_ids.shape[1] == 0 and prompt_str:
        print(f"Warning: Prompt '{prompt_str}' tokenized to an empty sequence. This may cause issues.")

    current_response_str = ""
    found_stop_sequence = None
    
    current_model_input_ids = full_input_ids # This will be moved to model_device in the loop

    for i in range(max_tokens):
        # Ensure input_ids are on the model's device for inference
        model_input_ids_on_device = current_model_input_ids.to(model_device)
        outputs = model(input_ids=model_input_ids_on_device) 
        
        logits_for_next_token = outputs.logits[:, -1, :] # Shape: (batch_size, vocab_size), on model_device

        # Calculate probabilities (on model_device)
        probs_unsorted = torch.softmax(logits_for_next_token, dim=-1)

        # Entropy check
        log_probs_unsorted = torch.log(probs_unsorted)
        entropy = -torch.sum(probs_unsorted * log_probs_unsorted.where(probs_unsorted > 0, torch.tensor(0.0, device=probs_unsorted.device)), dim=-1)
        print(f"\nEntropy: {entropy.item()}", flush=True)
        if entropy.item() > entropy_limit:
            print(f"\nEntropy {entropy.item()} exceeds limit {entropy_limit}. Stopping generation.")
            break

        # Sort original logits by value to get sorted_indices and sorted_logits.
        # sorted_indices contains original vocab indices, ordered by their logit values.
        # sorted_logits contains logits, ordered by their own values (descending).
        # Both are on model_device.
        sorted_logits, sorted_indices = torch.sort(logits_for_next_token, dim=-1, descending=True)

        # Prepare token strings for parse_fn, ordered by probability.
        # Assuming batch_size is 1 for this generation loop.
        # sorted_indices is (1, vocab_size), squeeze to (vocab_size,), then unsqueeze to (vocab_size, 1) for batch_true_next_token_mapping.
        # token_ids_for_parse_fn_sorted is on model_device; batch_true_next_token_mapping handles .cpu().
        token_ids_for_parse_fn_sorted = sorted_indices.squeeze(0).unsqueeze(-1) 
        all_token_strings_sorted = batch_true_next_token_mapping(
            token_ids_for_parse_fn_sorted, 
            tokenizer, 
            full_input_ids # Pass CPU tensor as context for tokenizer methods
        )

        # Call parse_fn with sorted token strings.
        # The returned mask (custom_mask_sorted) will correspond to this sorted order.
        # parse_fn is expected to return the mask on the device of `context_ids` (full_input_ids -> CPU).
        custom_mask_sorted = parse_fn(
            current_response_str, 
            all_token_strings_sorted, 
            tokenizer, 
            full_input_ids # Pass CPU tensor for context_ids, so mask is created on CPU
        )
        # custom_mask_sorted is now on CPU, shape (vocab_size,)

        # Ensure EOS token is allowed if it exists and is part of the sorted tokens.
        # custom_mask_sorted is on CPU. sorted_indices is on model_device.
        if tokenizer.eos_token_id is not None:
            # Bring sorted_indices (original vocab IDs) to CPU for comparison.
            # sorted_indices shape is (batch_size, vocab_size). Assuming batch_size=1.
            original_vocab_indices_sorted_cpu = sorted_indices.squeeze(0).cpu()
            
            # Find where the original EOS token ID appears in the sorted list of original IDs.
            eos_matches_in_sorted_list = (original_vocab_indices_sorted_cpu == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            
            if eos_matches_in_sorted_list.numel() > 0:
                idx_eos_in_sorted_list = eos_matches_in_sorted_list[0].item()
                # Ensure the found index is within the bounds of custom_mask_sorted.
                # This should always be true if custom_mask_sorted has length vocab_size.
                if idx_eos_in_sorted_list < custom_mask_sorted.shape[0]:
                    custom_mask_sorted[idx_eos_in_sorted_list] = True
                else:
                    # This case should ideally not be reached if vocab_size is consistent.
                    print(f"Warning: EOS token ID {tokenizer.eos_token_id} found at sorted index {idx_eos_in_sorted_list}, "
                          f"which is out of bounds for mask of size {custom_mask_sorted.shape[0]}. EOS might not be allowed.")
            # else: EOS token ID was not found among the tokens represented by sorted_indices.
            # This could happen if eos_token_id is outside the model's actual vocab range,
            # or has such low probability it's effectively not in sorted_indices (though sort covers all).

        # Initialize logits_to_sample_from with the sorted logits (on model_device)
        logits_to_sample_from = sorted_logits

        # Apply top_k if specified (operates on sorted logits)
        if top_k is not None and top_k > 0:
            effective_top_k = min(top_k, logits_to_sample_from.shape[-1])
            # Create a mask for the first effective_top_k elements in the sorted list
            top_k_indices_range = torch.arange(logits_to_sample_from.shape[-1], device=model_device)
            # top_k_on_sorted_mask is (vocab_size,), needs to be (1, vocab_size) for torch.where
            top_k_on_sorted_mask = (top_k_indices_range < effective_top_k).unsqueeze(0) 
            
            logits_to_sample_from = torch.where(
                top_k_on_sorted_mask, 
                logits_to_sample_from, 
                torch.tensor(-float("inf"), device=model_device)
            )

        # Apply min_p if specified (operates on potentially top_k'd sorted logits)
        if min_p > 0.0:
            # Calculate probabilities from current logits_to_sample_from (on model_device)
            # Softmax handles -inf correctly (they become 0 probability).
            current_probs_sorted = torch.softmax(logits_to_sample_from, dim=-1)
            min_p_on_sorted_mask = current_probs_sorted >= min_p # Shape (batch_size, vocab_size)
            logits_to_sample_from = torch.where(
                min_p_on_sorted_mask, 
                logits_to_sample_from, 
                torch.tensor(-float("inf"), device=model_device)
            )
        
        # Apply the custom grammar mask (custom_mask_sorted) to logits_to_sample_from.
        # custom_mask_sorted is on CPU, logits_to_sample_from is on model_device.
        # apply_mask will move custom_mask_sorted to model_device.
        masked_logits_sorted = apply_mask(logits_to_sample_from, custom_mask_sorted) # Result on model_device

        if (masked_logits_sorted == -float("inf")).all().item():
            print(f"\nWarning: All tokens masked out by combined filters for response prefix '{current_response_str}'. Stopping generation.")
            break
        
        # Apply temperature
        if temperature > 0.0: # Ensure float comparison
            masked_logits_sorted = masked_logits_sorted / temperature
        
        # Get probabilities from final masked & tempered logits (on model_device)
        masked_probs_sorted = torch.softmax(masked_logits_sorted, dim=-1)
        
        # Sample next token index (this index is relative to the sorted list)
        if temperature > 0.0: # Ensure float comparison
            next_token_relative_index = torch.multinomial(masked_probs_sorted, num_samples=1) # Shape (batch_size, 1), on model_device
        else: # Greedy decoding
            next_token_relative_index = torch.argmax(masked_logits_sorted, dim=-1, keepdim=True) # Shape (batch_size, 1), on model_device

        # Map the relative index back to the original vocabulary index.
        # sorted_indices is (batch_size, vocab_size), on model_device.
        # next_token_relative_index is (batch_size, 1), on model_device.
        next_token_original_vocab_index = sorted_indices.gather(-1, next_token_relative_index) # Shape (batch_size, 1), on model_device
        
        # Check for EOS token using the original vocabulary index
        if tokenizer.eos_token_id is not None and next_token_original_vocab_index.item() == tokenizer.eos_token_id:
            print("\nEOS token generated. Stopping.")
            break
        
        # Prepare the next token tensor for concatenation. Move to CPU (device of full_input_ids).
        next_token_tensor = next_token_original_vocab_index.to(device=full_input_ids.device, dtype=torch.int32) # Shape (batch_size, 1), on CPU

        # Decode the chosen token to string (for logging and current_response_str)
        # true_next_token_mapping expects prefix_ids on CPU if it uses .numpy() internally.
        token_text = true_next_token_mapping(next_token_tensor, tokenizer, full_input_ids) 
        
        current_response_str += token_text
        
        print("-"*10)
        print(current_response_str)
        print("-"*10, flush=True)
        
        if stop_sequences:
            for seq in stop_sequences:
                if current_response_str.endswith(seq):
                    found_stop_sequence = seq
                    print(f"\nStop sequence '{seq}' detected.", flush=True)
                    break 
            if found_stop_sequence:
                break
        
        # Append the new token (on CPU) to full_input_ids (on CPU)
        full_input_ids = torch.cat((full_input_ids, next_token_tensor), dim=1)
        current_model_input_ids = full_input_ids # Update for the next iteration (will be moved to model_device at loop start)
        
    return current_response_str, found_stop_sequence

