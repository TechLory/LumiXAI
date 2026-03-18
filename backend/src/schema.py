from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict
from typing import Any, List, Union, Optional
import numpy as np

class InputFeature(BaseModel):
    """Represents an atomic unit of input data.
    
    This class is modality-agnostic and serves as a standardized representation 
    for elements processed by the models.

    Attributes:
        index (Union[int, tuple]): The positional index of the feature. For text, this is an integer representing the token index. For images, it can be a tuple representing coordinates or regions.
        content (Any): The actual content of the feature (e.g., the string text of a token, or pixel values for an image patch).
        modality (str): The data modality, typically "text" or "image".
    """
    index: Union[int, tuple]
    content: Any
    modality: str  # "text", "image"

    def __repr__(self) -> str:
        """Returns a concise string representation for debugging.
        
        Returns:
            str: A formatted string containing the index, content, and modality of the feature.
        """
        return f"InputFeature(idx={self.index}, content='{self.content}', type={self.modality})"


class AttributionOutput(BaseModel):
    """Standardized output object returned by any Attributor.
    
    It serves as a contract between the backend (Attributors) and the frontend (GUI).
    It guarantees that visualizers always receive data in a predictable and consistent format.

    Attributes:
        heatmap (Any): The raw attribution scores. Can be a 1D array for text or higher-dimensional tensors/matrices for images.
        target (Any): The specific target that was explained (e.g., the predicted class index, or the generated token).
        input_features (List[InputFeature]): The list of input features corresponding to the heatmap scores. For text, its length should match the first dimension of the heatmap.
        generated_image (Optional[str]): A base64 encoded string of the generated image, applicable for image generation tasks. Defaults to None.
        metadata (dict): Additional contextual information or algorithm-specific metrics. Defaults to an empty dictionary.
        model_config (ConfigDict): Pydantic configuration allowing arbitrary types for fields like 'heatmap'.
    """
    
    # The raw attribution scores (heatmap). 
    heatmap: Any
    
    # The target that was explained (e.g., the class index or the generated token).
    target: Any
    
    # The list of input features corresponding to the heatmap scores.
    # Note: len(input_features) should ideally match heatmap.shape[0] for text.
    input_features: List[InputFeature]

    generated_image: Optional[str] = None

    metadata: dict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def validate(self) -> bool:
        """Validates the consistency of the attribution data.
        
        Helper method to check data consistency, particularly useful for debugging 
        pipeline errors ensuring the heatmap dimensions align with the input features.
        
        Returns:
            bool: True if dimensions match and the object is consistent.
            
        Raises:
            ValueError: If there is a dimension mismatch between the heatmap and the input features.
        """
        # Example validation for text modality (1D heatmap)
        if len(self.input_features) > 0 and self.input_features[0].modality == "text":
            if self.heatmap.ndim == 1 and len(self.input_features) != self.heatmap.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: Heatmap size {self.heatmap.shape[0]} "
                    f"!= Feature count {len(self.input_features)}"
                )
        return True