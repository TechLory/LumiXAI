from dataclasses import dataclass, field
from typing import Any, List, Union, Optional
import numpy as np

@dataclass
class InputFeature:
    """
    Represents an atomic unit of input data.
    
    This class is modality-agnostic:
    - For Text: Represents a token (index=int, content=string).
    - For Images: Represents a patch or pixel region (index=tuple/int, content=pixel_values).
    """
    index: Union[int, tuple]
    content: Any
    modality: str  # e.g., "text", "image"

    def __repr__(self) -> str:
        """Returns a concise string representation for debugging."""
        return f"InputFeature(idx={self.index}, content='{self.content}', type={self.modality})"


@dataclass
class AttributionOutput:
    """
    Standardized output object returned by any Attributor.
    
    It serves as a contract between the backend (Attributors) and the frontend (GUI).
    It guarantees that visualizers always receive data in a predictable format.
    """
    
    # The raw attribution scores (heatmap). 
    # For text: 1D array of shape [seq_len].
    # For images: 2D array of shape [height, width].
    heatmap: np.ndarray
    
    # The target that was explained (e.g., the class index or the generated token).
    target: Any
    
    # The list of input features corresponding to the heatmap scores.
    # Note: len(input_features) should ideally match heatmap.shape[0] for text.
    input_features: List[InputFeature]

    # Optional metadata dictionary for storing model-specific info (e.g., confidence scores).
    metadata: dict = field(default_factory=dict)

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