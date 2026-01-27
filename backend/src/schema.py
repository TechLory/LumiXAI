from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict
from typing import Any, List, Union, Optional
import numpy as np

class InputFeature(BaseModel):
    """
    Represents an atomic unit of input data.
    
    This class is modality-agnostic:
    - For Text: Represents a token (index=int, content=string).
    - For Images: Represents a patch or pixel region (index=tuple/int, content=pixel_values).
    """
    index: Union[int, tuple]
    content: Any
    modality: str  # "text", "image"

    def __repr__(self) -> str:
        """Returns a concise string representation for debugging."""
        return f"InputFeature(idx={self.index}, content='{self.content}', type={self.modality})"


class AttributionOutput(BaseModel):
    """
    Standardized output object returned by any Attributor.
    
    It serves as a contract between the backend (Attributors) and the frontend (GUI).
    It guarantees that visualizers always receive data in a predictable format.
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
        """
        Helper method to check data consistency.
        Useful for debugging pipeline errors.
        
        Returns:
            bool: True if dimensions match, raises ValueError otherwise.
        """
        # Example validation for text modality (1D heatmap)
        if len(self.input_features) > 0 and self.input_features[0].modality == "text":
            if self.heatmap.ndim == 1 and len(self.input_features) != self.heatmap.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: Heatmap size {self.heatmap.shape[0]} "
                    f"!= Feature count {len(self.input_features)}"
                )
        return True