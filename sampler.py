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
    return torch.ones(llm_tokenizer.vocab_size, dtype=torch.bool, device=context_ids.device)

def lambda_grammar_parse_fn(
    current_response_text: str, 
    all_token_strings: Tuple[str, ...], 
    llm_tokenizer: any,
    context_ids: torch.Tensor
) -> torch.Tensor:
    """
    A ParseFn that uses the typed_lambda_parser to constrain token generation
    based on the current response text, including type checking.
    """
    # Create a mask initialized to all False
    mask_np = np.zeros(llm_tokenizer.vocab_size, dtype=bool)
    
    # Normalize lambda characters in the current response
    normalized_response = current_response_text.replace('λ', 'lambda')
    
    # Process each token candidate
    for i, token_str in enumerate(all_token_strings):
        normalized_token = token_str.replace('λ', 'lambda')
        mask_np[i] = is_valid_token_candidate(normalized_response, normalized_token)
    
    # Always allow EOS token if available
    eos_token_id = llm_tokenizer.eos_token_id
    if eos_token_id is not None and eos_token_id < llm_tokenizer.vocab_size:
        mask_np[eos_token_id] = True
    
    # Log warning if no tokens are allowed by the grammar
    log_mask_warnings(mask_np, eos_token_id, current_response_text, all_token_strings)
    
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


def log_mask_warnings(mask_np: np.ndarray, eos_token_id: int, current_response_text: str, all_token_strings: Tuple[str, ...]) -> None:
    """Log warnings if the mask is too restrictive."""
    num_allowed = np.sum(mask_np)
    
    # Check if only EOS is allowed
    eos_only = (num_allowed == 1) and eos_token_id is not None and mask_np[eos_token_id]
    
    if num_allowed == 0:
        print(f"\nWarning: No tokens allowed by lambda grammar/type rules for: '{current_response_text}'. Generation will likely stop.")
    elif eos_only:
        print(f"\nWarning: No tokens allowed by lambda grammar/type rules for: '{current_response_text}'. Only EOS is permitted.")
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
) -> Tuple[str, str | None]:
    model_device_str = getattr(model, 'device', 'cpu')
    model_device = torch.device(model_device_str)
    
    print(f"Using device for model inference: {model_device}")

    model.eval()

    tokenized_prompt = tokenizer([prompt_str], return_tensors="pt") 
    full_input_ids = tokenized_prompt["input_ids"]

    if len(full_input_ids.shape) == 1:
        full_input_ids = full_input_ids.unsqueeze(0)
    
    if full_input_ids.shape[1] == 0 and prompt_str:
        print(f"Warning: Prompt '{prompt_str}' tokenized to an empty sequence by the wrapper. This may cause issues.")

    current_response_str = ""
    found_stop_sequence = None
    
    current_model_input_ids = full_input_ids

    for i in range(max_tokens):
        outputs = model(input_ids=current_model_input_ids) 
        
        logits_for_next_token = outputs.logits[:, -1, :] 

        probs = torch.softmax(logits_for_next_token, dim=-1)

        # Corrected entropy calculation to handle p=0 cases
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs.where(probs > 0, torch.tensor(0.0, device=probs.device)), dim=-1)
        print(f"\nEntropy: {entropy.item()}", flush=True)
        if entropy.item() > entropy_limit:
            print(f"\nEntropy {entropy.item()} exceeds limit {entropy_limit}. Stopping generation.")
            break

        vocab_indices_np = np.arange(tokenizer.vocab_size).reshape(-1, 1)
        vocab_indices = torch.tensor(vocab_indices_np, dtype=torch.int32, device=full_input_ids.device)
        all_token_strings = batch_true_next_token_mapping(vocab_indices, tokenizer, full_input_ids)

        custom_mask = parse_fn(current_response_str, all_token_strings, tokenizer, full_input_ids)
        
        masked_logits = apply_mask(logits_for_next_token, custom_mask)

        if (masked_logits == -float("inf")).all().item() > 0:
            print(f"\nWarning: All tokens masked out by parse_fn for response prefix '{current_response_str}'. Stopping generation.")
            break

        next_token_index = torch.argmax(masked_logits, dim=-1, keepdim=True)

        if tokenizer.eos_token_id is not None and next_token_index.item() == tokenizer.eos_token_id:
            print("\nEOS token generated. Stopping.")
            break
        
        next_token_tensor = next_token_index.to(full_input_ids.device).to(torch.int32)

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

        #print(token_text, end="", flush=True)

        full_input_ids = torch.cat((full_input_ids, next_token_tensor), dim=1)
        current_model_input_ids = full_input_ids
        
        if found_stop_sequence:
            break

    # Return only the newly generated response string and the stop sequence found (if any)
    return current_response_str, found_stop_sequence

