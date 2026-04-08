import torch
from typing import Any, Dict, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..abstract import BaseWrapper
from ..utils.hf_auth import hf_auth_kwargs

class HFTextClassificationWrapper(BaseWrapper):
    """Wrapper for Text Classification models (e.g., BERT, RoBERTa).
    
    This class utilizes Hugging Face's `AutoModelForSequenceClassification` 
    to handle standardized classification tasks, automatically managing 
    tokenization padding and inference logits extraction.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """Initializes the wrapper and triggers model loading.

        Args:
            model_id (str): The Hugging Face Hub ID or local path.
            device (str, optional): The target device ("cpu", "cuda", "mps"). Defaults to "cpu".
        """
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        """Loads the sequence classification model and its corresponding tokenizer.

        Automatically assigns the EOS token as the PAD token if the latter is missing, 
        ensuring compatibility with batch processing.

        Returns:
            Any: The loaded `AutoModelForSequenceClassification` PyTorch module.
        """
        print(f"Loading HF Classification Model: {self.model_id}...")
        auth_kwargs = hf_auth_kwargs()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **auth_kwargs)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **auth_kwargs)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        model.to(self.device)
        model.eval()
        return model

    def generate(self, input_data: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Performs a forward pass to compute classification logits.

        Args:
            input_data (Union[str, Dict[str, torch.Tensor]]): Either a raw text string 
                to be tokenized, or a dictionary of pre-tokenized tensors.

        Returns:
            torch.Tensor: The output logits tensor of shape `[Batch, NumClasses]`.
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
        
        return outputs.logits

    def get_embedding_layer(self) -> torch.nn.Module:
        """Retrieves the input embedding layer of the transformer.

        Returns:
            torch.nn.Module: The PyTorch embedding layer.
        """
        return self.model.get_input_embeddings()
