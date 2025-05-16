import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

# Imports from the project
from hf_wrappers import TransformersTokenizerWrapper, TransformersModelWrapper
# The following import is needed because hf_wrappers.py -> sampler.py -> typed_lambda_parser.py
# Ensure typed_lambda_parser.py and its dependencies (like Lark) are available.
import typed_lambda_parser # For side-effects if sampler.py needs it at import time

# --- Mock Hugging Face Output ---
class MockHFOutput:
    def __init__(self, logits: torch.Tensor, past_key_values: Optional[Any] = None):
        self.logits = logits
        self.past_key_values = past_key_values
class MockHFTokOutput: # Renamed for clarity
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


# --- Dummy Hugging Face Tokenizer (Refactored) ---
class DummyHfTokenizer:
    PAD_TOKEN = '<pad>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'

    def __init__(self, character_corpus: List[str]):
        self.special_tokens_map = {
            'pad_token': self.PAD_TOKEN,
            'eos_token': self.EOS_TOKEN,
            'unk_token': self.UNK_TOKEN,
            'bos_token': self.BOS_TOKEN,
        }

        # Build vocabulary list
        self._vocab_list = []
        self._vocab_list.extend([self.PAD_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN])
        
        unique_chars = sorted(list(set(c for c in character_corpus if c not in self._vocab_list)))
        self._vocab_list.extend(unique_chars)
        
        self._token_to_id_map = {token: i for i, token in enumerate(self._vocab_list)}
        
        # Set attributes expected by TransformersTokenizerWrapper and sampler
        self.pad_token = self.special_tokens_map['pad_token']
        self.eos_token = self.special_tokens_map['eos_token']
        self.unk_token = self.special_tokens_map['unk_token']
        self.bos_token = self.special_tokens_map['bos_token']

        self.pad_token_id = self._token_to_id_map[self.pad_token]
        self.eos_token_id = self._token_to_id_map[self.eos_token]
        self.unk_token_id = self._token_to_id_map[self.unk_token]
        self.bos_token_id = self._token_to_id_map[self.bos_token]
        
        self.vocab_size = len(self._vocab_list)

    def _tokenize_text_to_ids(self, text: str) -> List[int]:
        return [self._token_to_id_map.get(char, self.unk_token_id) for char in text]

    def __call__(self, text_input: Union[str, List[str]], return_tensors: str = "pt", 
                 padding: bool = False, truncation: bool = True, add_special_tokens: bool = False) -> MockHFTokOutput:
        if isinstance(text_input, str):
            text_list = [text_input]
        else:
            text_list = text_input
        
        text = text_list[0] # simplified_sampler and this test send one prompt at a time

        if not text: 
            token_ids = [self.bos_token_id]
        else:
            token_ids = self._tokenize_text_to_ids(text)

        if return_tensors == "pt":
            return MockHFTokOutput(torch.tensor([token_ids], dtype=torch.long))
        # For simplicity, always return PyTorch tensor in MockHFTokOutput for this dummy
        return MockHFTokOutput(torch.tensor([token_ids], dtype=torch.long))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # `add_special_tokens` is not implemented in this dummy version.
        return self._tokenize_text_to_ids(text)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        tokens = []
        for token_id in token_ids:
            if 0 <= token_id < len(self._vocab_list):
                token_str = self._vocab_list[token_id]
                if skip_special_tokens and token_str in self.special_tokens_map.values():
                    continue
                tokens.append(token_str)
            else:
                # Handle out-of-vocabulary token_id by appending unk_token representation
                if not (skip_special_tokens and self.unk_token in self.special_tokens_map.values()):
                    tokens.append(self.unk_token)
        return "".join(tokens)

    def batch_decode(self, list_of_token_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        return [self.decode(token_ids, skip_special_tokens=skip_special_tokens) for token_ids in list_of_token_ids]

    @property
    def all_special_tokens(self) -> List[str]:
        return list(self.special_tokens_map.values())

    @property
    def all_special_ids(self) -> List[int]:
        return [self._token_to_id_map[token] for token in self.all_special_tokens]

# --- Dummy Hugging Face Model ---
class DummyHfModel:
    def __init__(self, vocab_size: int, output_token_ids_sequence: List[int]):
        self.vocab_size = vocab_size
        self.output_token_ids_sequence = output_token_ids_sequence
        self.current_output_idx = 0
        self.device = torch.device("cpu") # Default device

    def __call__(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 past_key_values: Optional[Any] = None, use_cache: Optional[bool] = None) -> MockHFOutput:
        
        batch_size = input_ids.shape[0]
        
        if self.current_output_idx >= len(self.output_token_ids_sequence):
            next_token_id = DummyHfTokenizer([]).eos_token_id
        else:
            next_token_id = self.output_token_ids_sequence[self.current_output_idx]
        
        logits = torch.full((batch_size, input_ids.shape[1], self.vocab_size), -float('inf'), device=self.device)
        logits[:, -1, next_token_id] = 0.0  # High probability for the target token

        self.current_output_idx += 1
        
        return MockHFOutput(logits=logits, past_key_values=None)

    def eval(self):
        pass

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        return self

# --- Test Function ---
def test_infer_logic():
    print("\nStarting test_infer_logic...")

    user_query = "test query"
    think_content = "dummy think process"
    formal_content = "λx:A.x" # Contains the lambda character
    
    stop_think = "</think>"
    stop_formal = "</formal>"

    # This is a simplified representation of characters from the system prompt.
    # In a real scenario, ensure this covers all characters from the actual system prompt.
    system_prompt_example_chars = list("<system>You are an AI assistant. λx:(A->A).x</formal>")

    # Consolidate all characters expected in the test to form the tokenizer's character corpus
    character_corpus = list(set(
        list(user_query) +
        list(think_content) +
        list(formal_content) + # Ensures 'λ', 'x', ':', 'A', '.' are included
        list(stop_think) +
        list(stop_formal) +
        system_prompt_example_chars +
        list("12345().") # Other general characters
    ))
    
    dummy_tokenizer = DummyHfTokenizer(character_corpus)
    
    # Quick check that essential characters are in the vocab
    for char_val in list(formal_content) + ["<", ">", "/"]: # Check λ and tag characters
        assert char_val in dummy_tokenizer._token_to_id_map, f"Char '{char_val}' not in dummy vocab"
    assert dummy_tokenizer._token_to_id_map[dummy_tokenizer.bos_token] == dummy_tokenizer.bos_token_id

    think_phase_output_tokens = dummy_tokenizer.encode(think_content) + \
                                dummy_tokenizer.encode(stop_think)
    
    formal_phase_output_tokens = dummy_tokenizer.encode(formal_content) + \
                                 dummy_tokenizer.encode(stop_formal)

    full_model_output_sequence = think_phase_output_tokens + formal_phase_output_tokens

    dummy_model = DummyHfModel(dummy_tokenizer.vocab_size, full_model_output_sequence)

    wrapped_tokenizer = TransformersTokenizerWrapper(dummy_tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    wrapped_model = TransformersModelWrapper(dummy_model, device=device)

    print(f"Running wrapped_model.infer with user_query: '{user_query}'")
    generated_text = wrapped_model.infer(
        prompt_str=user_query,
        tokenizer=wrapped_tokenizer,
        max_think_tokens=len(think_content) + len(stop_think) + 5,
        max_formal_tokens=len(formal_content) + len(stop_formal) + 5,
        entropy_limit=10 
    )
    print(f"\nGenerated text from infer:\n---\n{generated_text}\n---")

    expected_user_tag_content = f"<user>{user_query}</user>"
    expected_think_tag_content = f"<think>{think_content}{stop_think}"
    expected_formal_tag_content = f"<formal>{formal_content}{stop_formal}"
    
    expected_output_str = (
        f"{expected_user_tag_content}\n"
        f"{expected_think_tag_content}\n"
        f"{expected_formal_tag_content}"
    )
    
    assert generated_text.strip() == expected_output_str.strip(), \
        f"Mismatch!\nExpected:\n---\n{expected_output_str.strip()}\n---\nGot:\n---\n{generated_text.strip()}\n---"

    print("\ntest_infer_logic PASSED!")

if __name__ == "__main__":
    from typing import Union
    try:
        test_infer_logic()
    except ImportError as e:
        print(f"ImportError: {e}. Ensure all project modules are accessible (e.g., via PYTHONPATH).")
        print("You might need to run this test from the root directory of the 'conai' project,")
        print("or ensure 'conai' and its submodules are in your Python path.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()