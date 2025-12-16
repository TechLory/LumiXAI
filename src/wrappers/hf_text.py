import torch
from typing import Any, Dict, Optional, Type
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    PreTrainedModel
)
from ..abstract import BaseWrapper

class HFTextWrapper(BaseWrapper):
    """
    Universal wrapper for Hugging Face Transformer models processing text.
    
    This class handles the initialization, tokenization, and forward pass 
    for various HF architectures (Classification, Causal LM, etc.) in a unified way.
    """

    # Mapping strategy to avoid extensive if-else chains.
    # We map specific architecture hints to the corresponding AutoModel class.
    # Future extension: Add 'ForTokenClassification', 'ForSeq2Seq', etc. here.
    MODEL_MAPPING: Dict[str, Type[PreTrainedModel]] = {
        "ForSequenceClassification": AutoModelForSequenceClassification,
        "ForCausalLM": AutoModelForCausalLM,
        # "ForConditionalGeneration": AutoModelForSeq2SeqLM, # Example for future extension
    }

    def load_model(self) -> PreTrainedModel:
        """
        Loads the tokenizer and the model based on the model_id.
        Automatically detects the task type (classification vs generation) 
        from the model configuration.

        Returns:
            PreTrainedModel: The loaded PyTorch model set to evaluation mode.
        """
        # Suggestion: If this block grows, move config loading to a separate _load_config() method.
        try:
            config = AutoConfig.from_pretrained(self.model_id)
        except OSError as e:
            raise ValueError(f"Could not load configuration for {self.model_id}: {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Ensure padding token exists (critical for GPT-like models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine the correct model class using the registry pattern
        model_class = self._determine_model_class(config)
        
        # Load the actual model weights
        model = model_class.from_pretrained(self.model_id)
        model.to(self.device)
        model.eval()
        
        return model

    def _determine_model_class(self, config: Any) -> Type[PreTrainedModel]:
        """
        Helper method to select the correct AutoModel class based on config architectures.
        
        Args:
            config: The AutoConfig object loaded from Hugging Face.
            
        Returns:
            The class type (e.g., AutoModelForSequenceClassification).
        """
        # Retrieve the list of architectures defined in config.json
        archs = getattr(config, "architectures", [])
        
        for arch in archs:
            for key, model_cls in self.MODEL_MAPPING.items():
                if key in arch:
                    return model_cls
        
        # Fallback: If unknown, default to CausalLM (Generative) or raise specific error
        return AutoModelForCausalLM

    def generate(self, input_data: str) -> Dict[str, Any]:
        """
        Performs tokenization and a forward pass to retrieve logits.
        
        Args:
            input_data (str): The raw input text.

        Returns:
            Dict[str, Any]: A dictionary containing logits, input_ids, attention_mask, 
                            and decoded tokens for visualization.
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_data, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        # Perform forward pass (gradients are kept enabled for Captum compatibility)
        outputs = self.model(**inputs)

        return {
            "logits": outputs.logits,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "tokens": self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        }

    def get_embedding_layer(self) -> torch.nn.Module:
        """
        Identifies and returns the input embedding layer of the model.
        Required for gradient-based attribution methods (e.g., Integrated Gradients).
        
        Returns:
            torch.nn.Module: The embedding layer object.
            
        Raises:
            ValueError: If the embedding layer cannot be found automatically.
        """
        model = self.model
        
        # List of common attribute names for embeddings in HF models
        candidate_names = ["embeddings", "wte", "word_embeddings", "shared"]

        # Strategy 1: Check the top-level model
        for name in candidate_names:
            if hasattr(model, name):
                return getattr(model, name)
            
        # Strategy 2: Check inside the base model wrapper (e.g., model.bert)
        # Suggestion: If this logic gets more complex, consider a recursive search function.
        if hasattr(model, "base_model_prefix"):
            base_model_name = model.base_model_prefix
            if hasattr(model, base_model_name):
                base_model = getattr(model, base_model_name)
                for name in candidate_names:
                    if hasattr(base_model, name):
                        return getattr(base_model, name)

        raise ValueError(f"Could not automatically locate embedding layer for {self.model_id}.")