import torch
from typing import Any, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..abstract import BaseWrapper

class HFTextGenerationWrapper(BaseWrapper):
    """
    Wrapper for Causal Language Models (e.g., GPT-2, Llama, Gemma).
    Uses AutoModelForCausalLM from HuggingFace.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        print(f"Loading HF Generation Model: {self.model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        model.to(self.device)
        model.eval()
        return model

    def generate(self, input_data: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Input: Text string or tokenized inputs
        Output: Logits for next token [Batch, VocabSize]
        """
        # If a string is provided, tokenize it
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
        
        # Extract logits only for the last token in the sequence
        # outputs.logits shape is [Batch, SeqLen, VocabSize]
        next_token_logits = outputs.logits[:, -1, :]
        
        return next_token_logits

    def get_embedding_layer(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()