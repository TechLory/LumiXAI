from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
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