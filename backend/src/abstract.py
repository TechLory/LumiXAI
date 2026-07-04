from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch

# Import schema to enforce return types
from .schema import AttributionOutput

class BaseWrapper(ABC):
    """Abstract Base Class for all Model Wrappers.
    
    A Wrapper acts as an adapter around a specific model architecture (e.g., Hugging Face 
    Transformers, Diffusers), providing a unified interface for loading, inference, 
    and internal state access.
    """

    def __init__(self, model_id: str, device: str):
        """Initializes the wrapper configuration and loads the model.

        Args:
            model_id (str): The identifier for the model (e.g., a Hugging Face Hub ID or a local directory path).
            device (str): The execution device to load the model onto (e.g., 'cpu', 'cuda', 'mps').
        """
        self.model_id = model_id
        self.device = device
        
        # Automatically load the model upon initialization
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        """Loads the model into memory.
        
        This is an abstract method and must be implemented by concrete subclasses 
        to handle the specific instantiation logic of the underlying framework.

        Returns:
            Any: The loaded model object (e.g., a torch.nn.Module or a Pipeline).
        """
        pass

    @abstractmethod
    def generate(self, input_data: Any) -> Any:
        """Performs a forward pass or generation step.
        
        Args:
            input_data (Any): The input payload required by the model to perform generation or inference.
            
        Returns:
            Any: The raw output from the model (e.g., logits for text classification, or a PIL Image for visual generation).
        """
        pass

    def get_embedding_layer(self) -> Any:
        """Retrieves the input embedding layer of the model. 
        
        Crucial for gradient-based attribution methods (e.g., Captum Integrated Gradients) 
        which require gradients to be computed with respect to the continuous embeddings.
        This method is not abstract because not all models utilize standard embeddings, 
        but it must be overridden if a specific attributor requires it.

        Returns:
            Any: The embedding layer object (e.g., torch.nn.Embedding).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"The wrapper for {self.model_id} does not support embedding retrieval, "
            "or the method 'get_embedding_layer' has not been implemented."
        )

    def get_special_tokens_mask(self, input_ids: Any) -> List[bool]:
        """Returns a boolean mask flagging structural/special tokens (e.g. ``[CLS]``, ``[SEP]``, BOS/EOS).

        This is purely tokenizer metadata and does NOT alter attribution scores; it lets the
        frontend optionally hide special tokens from the visualization so that heatmaps do not
        "sink" into structural tokens. Like ``get_embedding_layer``, this method is intentionally
        not abstract: not every modality has a tokenizer or a notion of special tokens. The default
        implementation opportunistically uses ``self.tokenizer`` when present, and otherwise treats
        every feature as non-special.

        Args:
            input_ids (Any): A 1D tensor or an iterable of token ids aligned with the input features.

        Returns:
            List[bool]: One flag per token; True where the token is a special/structural token.
        """
        ids = input_ids.detach().cpu().tolist() if hasattr(input_ids, "detach") else list(input_ids)
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return [False] * len(ids)
        try:
            mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            return [bool(m) for m in mask]
        except Exception:
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            return [tid in special_ids for tid in ids]

    def get_template_tokens_mask(self, text: str, input_ids: Any) -> List[bool]:
        """Returns a boolean mask flagging chat-template / structural scaffolding tokens.

        This is a separate category from :meth:`get_special_tokens_mask`: it targets the
        tokens a chat template inserts *around* the user content (role markers such as
        ``user``/``assistant``, control tokens like ``<|im_start|>``, and formatting),
        which also act as attention sinks. Like the other metadata helpers, it never alters
        attribution scores and is intentionally non-abstract: models without a chat template
        (e.g. classification, plain GPT-2) fall back to the default here, which treats every
        token as non-template.

        Args:
            text (str): The raw user content that was wrapped by the template.
            input_ids (Any): A 1D tensor or iterable of token ids aligned with the input features.

        Returns:
            List[bool]: One flag per token; True where the token is template scaffolding.
        """
        ids = input_ids.detach().cpu().tolist() if hasattr(input_ids, "detach") else list(input_ids)
        return [False] * len(ids)


class BaseAttributor(ABC):
    """Abstract Base Class for all Attribution Methods.
    
    An Attributor takes a Wrapper (the model) and explains its predictions.
    It encapsulates the core logic of Explainable AI algorithms such as 
    Integrated Gradients, DAAM, or SHAP.
    """

    def __init__(self, wrapper: BaseWrapper):
        """Initializes the attributor via dependency injection.
        
        Args:
            wrapper (BaseWrapper): An instantiated concrete Wrapper subclass representing the loaded model.
        """
        self.wrapper = wrapper

    @abstractmethod
    def attribute(self, input_data: Any, target_output: Any = None, **kwargs) -> AttributionOutput:
        """Calculates feature importance scores for a given input.

        Args:
            input_data (Any): The raw input provided to the model that needs to be explained.
            target_output (Optional[Any], optional): The specific output or class index to explain. 
                If None, the algorithm defaults to explaining the model's top prediction. Defaults to None.
            **kwargs: Additional algorithm-specific parameters (e.g., number of steps for Integrated Gradients).

        Returns:
            AttributionOutput: A standardized result object containing the heatmaps and input features.
        """
        pass