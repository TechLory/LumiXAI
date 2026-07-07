import torch
from typing import Any
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from ..abstract import BaseWrapper
from ..utils.hf_auth import hf_auth_kwargs

class HFImageClassificationWrapper(BaseWrapper):
    """Wrapper for Image Classification models (e.g., ViT, ResNet, ConvNeXt).

    Uses Hugging Face's `AutoModelForImageClassification` together with the model's own
    `AutoImageProcessor` to standardize preprocessing (resize, normalization) across
    architectures. Unlike the text wrappers, attribution methods operate directly on the
    normalized pixel tensor rather than an intermediate embedding layer: pixel values are
    already continuous and differentiable, so no analogue to `get_embedding_layer` is
    needed for gradient-based attribution here.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """Initializes the wrapper and triggers model loading.

        Args:
            model_id (str): The Hugging Face Hub ID or local path.
            device (str, optional): The target device ("cpu", "cuda", "mps"). Defaults to "cpu".
        """
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        """Loads the image classification model and its corresponding image processor.

        Returns:
            Any: The loaded `AutoModelForImageClassification` PyTorch module.
        """
        print(f"Loading HF Image Classification Model: {self.model_id}...")
        auth_kwargs = hf_auth_kwargs()

        self.processor = AutoImageProcessor.from_pretrained(self.model_id, **auth_kwargs)

        model = AutoModelForImageClassification.from_pretrained(self.model_id, **auth_kwargs)
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, input_data: Image.Image) -> torch.Tensor:
        """Converts a PIL Image into the model's normalized pixel tensor.

        Args:
            input_data (PIL.Image.Image): The raw input image.

        Returns:
            torch.Tensor: Pixel tensor of shape `[1, C, H, W]` on the wrapper's device.
        """
        if not isinstance(input_data, Image.Image):
            raise ValueError("Input for HFImageClassificationWrapper must be a PIL Image.")

        inputs = self.processor(images=input_data, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def generate(self, input_data: Image.Image) -> torch.Tensor:
        """Performs a forward pass to compute classification logits.

        Args:
            input_data (PIL.Image.Image): The image to classify.

        Returns:
            torch.Tensor: The output logits tensor of shape `[Batch, NumClasses]`.
        """
        pixel_values = self.preprocess(input_data)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        return outputs.logits

    def get_predicted_label(self, class_id: int) -> str:
        """Resolves a predicted class index to its human-readable label.

        Args:
            class_id (int): The predicted class index.

        Returns:
            str: The label from `model.config.id2label`, or the raw index as a string
                if no mapping is available.
        """
        id2label = getattr(self.model.config, "id2label", None) or {}
        return id2label.get(class_id, str(class_id))
