import torch
from typing import Callable, Tuple, List, Dict, Any
import numpy as np

import typed_lambda_parser
from typed_lambda_parser import TypeCheckError, ParserSyntaxError

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
    mask_np = np.zeros(llm_tokenizer.vocab_size, dtype=bool)

    # Normalize potential lambda characters in the current response once
    _current_response_text_normalized = current_response_text.replace('λ', 'lambda')

    for i, token_candidate_str in enumerate(all_token_strings):
        # Normalize potential lambda characters in the candidate token
        _token_candidate_str_normalized = token_candidate_str.replace('λ', 'lambda')
        print(f"Checking token candidate: '{_token_candidate_str_normalized}' against current response: '{_current_response_text_normalized}'")
        
        # test_expr_str is the current response being built + the candidate token string (both normalized)
        test_expr_str = _current_response_text_normalized + _token_candidate_str_normalized
        
        try:
            if not test_expr_str.strip():
                if any(c.isspace() for c in _token_candidate_str_normalized) and _token_candidate_str_normalized:
                    mask_np[i] = True
                continue

            lambda_tokens = typed_lambda_parser.tokenize(test_expr_str)
            if not lambda_tokens and test_expr_str.strip():
                pass
            elif not lambda_tokens and not test_expr_str.strip():
                pass
            else:
                parser = typed_lambda_parser.Parser(lambda_tokens)
                ast = parser.parse_expression()
                
                try:
                    typed_lambda_parser.infer_type(ast, {}) 
                    mask_np[i] = True
                except TypeCheckError:
                    mask_np[i] = False
                    
        except ParserSyntaxError as e:
            is_prefix = False
            if e.parser_instance is not None:
                local_parser_instance = e.parser_instance
                if local_parser_instance.pos >= len(local_parser_instance.tokens) - 1:
                    is_prefix = True
            
            if is_prefix:
                mask_np[i] = True
            elif not _current_response_text_normalized.strip():
                try:
                    candidate_lambda_tokens = typed_lambda_parser.tokenize(_token_candidate_str_normalized)
                    if candidate_lambda_tokens:
                        if candidate_lambda_tokens[0].type in ('IDENTIFIER', 'LAMBDA', 'LPAREN'):
                            mask_np[i] = True
                except ParserSyntaxError:
                    pass
        except Exception:
            pass

    num_allowed_by_grammar_rules = np.sum(mask_np)

    eos_token_id = llm_tokenizer.eos_token_id
    eos_added_to_mask = False
    if eos_token_id is not None and eos_token_id < llm_tokenizer.vocab_size:
        if not mask_np[eos_token_id]:
            eos_added_to_mask = True
        mask_np[eos_token_id] = True
    
    mask = torch.tensor(mask_np, dtype=torch.bool, device=context_ids.device)
    final_allowed_count = mask.sum().item()

    if num_allowed_by_grammar_rules == 0:
        if final_allowed_count == 1 and eos_added_to_mask:
            print(f"\nWarning: No tokens allowed by lambda grammar/type rules for response prefix: '{current_response_text}'. Only EOS is now permitted.")
        elif final_allowed_count == 0:
            print(f"\nWarning: No tokens allowed by lambda grammar/type rules for response prefix: '{current_response_text}' and EOS is not available/effective. Generation will likely stop.")
        else:
            print(f"\nWarning: Lambda grammar/type rules themselves allowed 0 tokens for response prefix: '{current_response_text}'. Current mask allows {final_allowed_count} token(s) (EOS may be included).")

    return mask


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

        # plot the probs in a graph
        print(f"\nLogits for next token: {logits_for_next_token}")
        print(f"\nProbabilities for next token: {probs}", flush=True)



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
        print(all_token_strings)
        print(current_response_str)
        custom_mask = parse_fn(current_response_str, all_token_strings, tokenizer, full_input_ids)

        print(custom_mask)
        
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
        
        if stop_sequences:
            for seq in stop_sequences:
                if current_response_str.endswith(seq):
                    found_stop_sequence = seq
                    print(f"\nStop sequence '{seq}' detected.", flush=True)
                    break 
            if found_stop_sequence:
                break

        print(token_text, end="", flush=True)

        full_input_ids = torch.cat((full_input_ids, next_token_tensor), dim=1)
        current_model_input_ids = full_input_ids
        
        if found_stop_sequence:
            break

    # Return only the newly generated response string and the stop sequence found (if any)
    return current_response_str, found_stop_sequence

