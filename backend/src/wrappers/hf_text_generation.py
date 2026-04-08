import torch
import torch.nn.functional as F
from typing import Any, Dict, Union, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..abstract import BaseWrapper
from ..utils.hf_auth import hf_auth_kwargs

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

    def generate_text(self, prompt: str, max_new_tokens: int = 20) -> Tuple[str, List[str], List[float]]:
        """Generates a sequence of text deterministically and extracts token probabilities.

        Args:
            prompt (str): The initial text context.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 20.

        Returns:
            Tuple[str, List[str], List[float]]: A tuple containing:
                - The full generated text string (excluding the prompt).
                - A list of individual generated token strings.
                - A list of float probabilities (0.0 - 1.0) for each generated token.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False, 
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_ids = output_ids[0][input_length:]
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        probs = []
        tokens_text = []

        current_input_ids = inputs.input_ids        
        for token_id in generated_ids:
            with torch.no_grad():
                outputs = self.model(current_input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                token_prob = next_token_probs[0, token_id].item()
                probs.append(token_prob)
                
                token_str = self.tokenizer.decode([token_id])
                tokens_text.append(token_str)
                
                current_input_ids = torch.cat([current_input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)

        return full_text, tokens_text, probs

    def generate(self, input_data: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Performs a single forward pass to extract next-token logits.

        Args:
            input_data (Union[str, Dict[str, torch.Tensor]]): Raw text or tokenized dictionary.

        Returns:
            torch.Tensor: Logits for the next token prediction, shape `[Batch, VocabSize]`.
        """
        if isinstance(input_data, str):
            inputs = self.tokenizer(
                input_data, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
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
