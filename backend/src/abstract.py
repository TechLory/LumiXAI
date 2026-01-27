from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch

# Import schema to enforce return types
from .schema import AttributionOutput

class BaseWrapper(ABC):
    """
    Abstract Base Class for all Model Wrappers.
    
    A Wrapper acts as an adapter around a specific model architecture (e.g., HF Transformers, Diffusers),
    providing a unified interface for loading, inference, and internal state access.
    """

    def __init__(self, model_id: str, device: str):
        """
        Initializes the wrapper configuration.

        Args:
            model_id (str): The identifier for the model (e.g., Hugging Face ID or local path).
            device (str): The execution device ('cpu', 'cuda', 'mps').
        """
        self.model_id = model_id
        self.device = device
        
        # Automatically load the model upon initialization
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        """
        Abstract method to load the model into memory.
        Must be implemented by the concrete subclass.

        Returns:
            Any: The loaded model object (e.g., torch.nn.Module).
        """
        pass

    @abstractmethod
    def generate(self, input_data: Any) -> Any:
        """
        Performs a forward pass or generation step.
        Returns raw output (Logits for text, PIL Image for visual).
        """
        pass

    def get_embedding_layer(self) -> Any:
        """
        Retrieves the input embedding layer. 
        Crucial for gradient-based attribution methods (e.g., Captum).
        
        This method is not abstract because not all models use embeddings.
        However, if an Attributor requires it, the concrete Wrapper must implement it.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"The wrapper for {self.model_id} does not support embedding retrieval, "
            "or the method 'get_embedding_layer' has not been implemented."
        )


class BaseAttributor(ABC):
    """
    Abstract Base Class for all Attribution Methods.
    
    An Attributor takes a Wrapper (the model) and explains its predictions.
    It encapsulates the logic of algorithms like Integrated Gradients, DAAM, or SHAP.
    """

    def __init__(self, wrapper: BaseWrapper):
        """
        Dependency Injection: The Attributor receives a ready-to-use model wrapper.
        
        Args:
            wrapper (BaseWrapper): An instance of a concrete Wrapper subclass.
        """
        self.wrapper = wrapper

    @abstractmethod
    def attribute(self, input_data: Any, target_output: Any = None) -> AttributionOutput:
        """
        Calculates feature importance scores.

        Args:
            input_data (Any): The raw input to explain.
            target_output (Any, optional): The specific output/class to explain. 
                                           If None, defaults to the model's prediction.

        Returns:
            AttributionOutput: The standardized result object.
        """
        pass