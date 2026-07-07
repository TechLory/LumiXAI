import torch
from typing import Any
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from ..abstract import BaseWrapper
from ..utils.hf_auth import hf_auth_kwargs
from ..utils.image_attribution import denormalize_pixel_values

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

    def get_display_image(self, pixel_values: torch.Tensor) -> Image.Image:
        """Reconstructs the exact image the model saw from its normalized pixel tensor.

        Attributions live in the processor's normalized pixel space, so this de-normalized
        image (not the raw upload) is the correct background for the heatmap overlay: it
        stays aligned with the attribution grid no matter how the processor resized,
        cropped, or squished the original. Pulls the normalization constants straight from
        the model's own processor, so it works for any loaded model without configuration.

        Args:
            pixel_values (torch.Tensor): The `[1, C, H, W]` tensor from `preprocess`.

        Returns:
            PIL.Image.Image: The de-normalized image the model actually received.
        """
        normalizes = getattr(self.processor, "do_normalize", True)
        image_mean = getattr(self.processor, "image_mean", None) if normalizes else None
        image_std = getattr(self.processor, "image_std", None) if normalizes else None
        return denormalize_pixel_values(pixel_values, image_mean, image_std)

    def get_gradcam_layer(self, pixel_values: torch.Tensor) -> torch.nn.Module:
        """Finds the target layer for Grad-CAM generically, for any loaded architecture.

        Grad-CAM needs a layer that still has spatial structure (a `[B, C, H, W]` feature
        map). Rather than hardcode per-model layer paths — which would break the moment an
        unfamiliar model is loaded — this runs one forward pass with hooks and returns the
        *last* module (in execution order) whose output is a 4D tensor with both spatial
        dims > 1. That single heuristic lands on the right layer for both families:

        - **CNNs** (ResNet/ConvNeXt): the deepest convolutional feature map (e.g. 7x7),
          the canonical Grad-CAM target.
        - **ViTs**: the patch-embedding projection's `[B, hidden, 14, 14]` output — the
          only 4D map, since everything after it is a flattened `[B, seq, hidden]` token
          sequence. This yields clean per-patch attribution instead of pixel-grid artifacts.

        Args:
            pixel_values (torch.Tensor): A `[1, C, H, W]` tensor from `preprocess`.

        Returns:
            torch.nn.Module: The module to attach Grad-CAM to.

        Raises:
            RuntimeError: If no spatial feature map is found (e.g. a non-convolutional,
                non-patch model Grad-CAM cannot meaningfully explain).
        """
        candidates: list[torch.nn.Module] = []
        handles = []

        def hook(module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 4 and tensor.shape[2] > 1 and tensor.shape[3] > 1:
                candidates.append(module)

        for module in self.model.modules():
            if module is not self.model:
                handles.append(module.register_forward_hook(hook))

        try:
            with torch.no_grad():
                self.model(pixel_values=pixel_values)
        finally:
            for handle in handles:
                handle.remove()

        if not candidates:
            raise RuntimeError(
                f"Could not locate a spatial (4D) feature map in {self.model_id} for Grad-CAM; "
                "this model may not be convolutional or patch-based."
            )
        # Hooks fire in execution order, and a container fires after its children, so the
        # last candidate is the deepest spatial feature map in the forward pass.
        return candidates[-1]

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
