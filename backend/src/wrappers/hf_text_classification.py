import torch
from typing import Any, Dict, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..abstract import BaseWrapper

class HFTextClassificationWrapper(BaseWrapper):
    """
    Wrapper for Text Classification models (e.g., BERT, RoBERTa).
    Uses AutoModelForSequenceClassification from HuggingFace.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        print(f"Loading HF Classification Model: {self.model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        model.to(self.device)
        model.eval()
        return model

    def generate(self, input_data: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Input: Text string or tokenized inputs
        Output: Logits [Batch, NumClasses]
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
        
        return outputs.logits

    def get_embedding_layer(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()