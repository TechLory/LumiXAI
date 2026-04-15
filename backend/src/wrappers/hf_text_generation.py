import os
import torch
import torch.nn.functional as F
from typing import Any, Dict, Union, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..abstract import BaseWrapper
from ..utils.hf_auth import hf_auth_kwargs

TEXT_MAX_NEW_TOKENS_ENV_VAR = "LUMIXAI_TEXT_MAX_NEW_TOKENS"
DEFAULT_TEXT_MAX_NEW_TOKENS = 64


def get_default_text_max_new_tokens() -> int:
    raw_value = os.getenv(TEXT_MAX_NEW_TOKENS_ENV_VAR, str(DEFAULT_TEXT_MAX_NEW_TOKENS)).strip()

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return DEFAULT_TEXT_MAX_NEW_TOKENS

    return parsed_value if parsed_value > 0 else DEFAULT_TEXT_MAX_NEW_TOKENS


def get_model_max_positions(model: Any) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None

    for attr in ("max_position_embeddings", "n_positions"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    return None


def normalize_eos_token_ids(token_id_config: Any) -> set[int]:
    if token_id_config is None:
        return set()
    if isinstance(token_id_config, int):
        return {token_id_config}
    if isinstance(token_id_config, (list, tuple, set)):
        return {int(token_id) for token_id in token_id_config}
    return set()


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }

class HFTextGenerationWrapper(BaseWrapper):
    """Wrapper for Causal Language Models (e.g., GPT-2, Llama, Qwen).
    
    This class utilizes Hugging Face's `AutoModelForCausalLM` and implements specialized 
    methods for autoregressive decoding, specifically designed to extract step-by-step 
    probabilities and logits required for XAI attribution loops.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """Initializes the wrapper and triggers model loading.

        Args:
            model_id (str): The Hugging Face Hub ID or local path.
            device (str, optional): The target device ("cpu", "cuda", "mps"). Defaults to "cpu".
        """
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        """Loads the causal language model and its corresponding tokenizer.

        Returns:
            Any: The loaded `AutoModelForCausalLM` PyTorch module.
        """
        print(f"Loading HF Generation Model: {self.model_id}...")
        auth_kwargs = hf_auth_kwargs()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **auth_kwargs)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **auth_kwargs)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        model.to(self.device)
        model.eval()
        return model

    def has_chat_template(self) -> bool:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        return isinstance(chat_template, str) and bool(chat_template.strip())

    def tokenize_generation_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        if self.has_chat_template():
            messages = [{"role": "user", "content": prompt}]
            try:
                batch = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }

            if hasattr(batch, "to"):
                return batch.to(self.device)
            return move_batch_to_device(batch, self.device)

        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def get_generation_eos_token_id(self) -> int | List[int] | None:
        eos_token_ids = normalize_eos_token_ids(
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        )

        if not eos_token_ids:
            eos_token_ids = normalize_eos_token_ids(getattr(self.model.config, "eos_token_id", None))

        if not eos_token_ids:
            eos_token_ids = normalize_eos_token_ids(getattr(self.tokenizer, "eos_token_id", None))

        if not eos_token_ids:
            return None

        ordered_ids = sorted(eos_token_ids)
        if len(ordered_ids) == 1:
            return ordered_ids[0]

        return ordered_ids

    def generate_text(self, prompt: str, max_new_tokens: int | None = None) -> Tuple[str, List[int], List[str], List[float]]:
        """Generates a sequence of text deterministically and extracts token probabilities.

        Args:
            prompt (str): The initial text context.
            max_new_tokens (int | None, optional): The maximum number of tokens to generate.
                If omitted, the backend uses `LUMIXAI_TEXT_MAX_NEW_TOKENS` or falls back to 64.

        Returns:
            Tuple[str, List[int], List[str], List[float]]: A tuple containing:
                - The full generated text string (excluding the prompt).
                - A list of the generated token ids.
                - A list of individual generated token strings.
                - A list of float probabilities (0.0 - 1.0) for each generated token.
        """
        inputs = self.tokenize_generation_prompt(prompt)
        resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else get_default_text_max_new_tokens()
        input_length = inputs["input_ids"].shape[1]
        max_positions = get_model_max_positions(self.model)
        eos_token_id = self.get_generation_eos_token_id()

        if max_positions is not None:
            available_new_tokens = max_positions - input_length
            if available_new_tokens <= 0:
                raise ValueError(
                    f"Prompt is too long for model '{self.model_id}'. "
                    f"Input tokens: {input_length}, model context window: {max_positions}."
                )

            if resolved_max_new_tokens > available_new_tokens:
                print(
                    f"Clamping max_new_tokens from {resolved_max_new_tokens} to {available_new_tokens} "
                    f"for model '{self.model_id}' (context window: {max_positions}, input tokens: {input_length})."
                )
                resolved_max_new_tokens = available_new_tokens

        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": resolved_max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id

            output_ids = self.model.generate(**generation_kwargs)
        
        generated_ids = output_ids[0][input_length:]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        probs = []
        tokens_text = []
        generated_token_ids = [int(token_id.item()) for token_id in generated_ids]

        current_input_ids = inputs["input_ids"]
        for token_id in generated_ids:
            with torch.no_grad():
                outputs = self.model(current_input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                token_prob = next_token_probs[0, token_id].item()
                probs.append(token_prob)
                
                token_str = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                tokens_text.append(token_str)
                
                current_input_ids = torch.cat([current_input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)

        return full_text, generated_token_ids, tokens_text, probs

    def generate(self, input_data: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Performs a single forward pass to extract next-token logits.

        Args:
            input_data (Union[str, Dict[str, torch.Tensor]]): Raw text or tokenized dictionary.

        Returns:
            torch.Tensor: Logits for the next token prediction, shape `[Batch, VocabSize]`.
        """
        if isinstance(input_data, str):
            inputs = self.tokenize_generation_prompt(input_data)
        else:
            inputs = input_data

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        next_token_logits = outputs.logits[:, -1, :]
        
        return next_token_logits
    
    def forward_func(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor: # type: ignore
        """Adapter forward function required by Captum Integrated Gradients.

        Bypasses the standard tokenizer input and processes continuous embedding tensors directly.

        Args:
            input_embeds (torch.Tensor): Continuous input embeddings.
            attention_mask (torch.Tensor, optional): Standard attention mask. Defaults to None.

        Returns:
            torch.Tensor: Logits of the last token (next-token prediction).
        """
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.logits[:, -1, :]

    def get_embedding_layer(self) -> torch.nn.Module:
        """Retrieves the input embedding layer of the causal LM.

        Returns:
            torch.nn.Module: The PyTorch embedding layer.
        """
        return self.model.get_input_embeddings()
