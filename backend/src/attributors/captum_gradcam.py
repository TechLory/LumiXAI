import torch
from typing import Optional, Any
from PIL import Image
from captum.attr import LayerGradCam, LayerAttribution

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_image_classification import HFImageClassificationWrapper
from ..utils.image_attribution import render_image_heatmap, image_to_base64, decode_base64_image


class CaptumGradCamAttributor(BaseAttributor):
    """Image-classification attributor: Grad-CAM via Captum's LayerGradCam.

    Where the other attributors work in raw pixel space (and are therefore inherently
    grainy), Grad-CAM attributes at a deep *spatial feature map* and upsamples the result,
    producing the smooth, object-following heatmaps Grad-CAM is known for. The target layer
    is chosen generically at runtime (see `HFImageClassificationWrapper.get_gradcam_layer`),
    so this resolves to the last conv map for CNNs and the patch-embedding grid for ViTs —
    no per-model configuration, matching the "works for any loaded model" constraint.

    Image classification only; registered with `hf_image_classification` as its sole
    compatible wrapper.
    """

    def attribute(self, input_data: Any, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Computes a Grad-CAM heatmap for an image classification model.

        Args:
            input_data (Any): A PIL Image or a base64-encoded image string.
            target_output (Optional[int], optional): The target class index. If None, the
                predicted class is used.

        Returns:
            AttributionOutput: A single-entry pixel heatmap mapping image regions to importance.
        """
        if not isinstance(self.wrapper, HFImageClassificationWrapper):
            raise NotImplementedError(
                "Grad-CAM (Captum) is only available for image classification models."
            )

        wrapper = self.wrapper
        image = input_data if isinstance(input_data, Image.Image) else decode_base64_image(input_data)
        pixel_values = wrapper.preprocess(image)

        def model_forward(pixels):
            return wrapper.model(pixel_values=pixels).logits

        if target_output is None:
            with torch.no_grad():
                logits = wrapper.model(pixel_values=pixel_values).logits
            target_output = torch.argmax(logits, dim=1).item()

        layer = wrapper.get_gradcam_layer(pixel_values)
        gradcam = LayerGradCam(model_forward, layer)

        # `relu_attributions=True` keeps only evidence *for* the target class, the classic
        # Grad-CAM formulation. The result is a coarse `[1, 1, Hf, Wf]` map (the feature-map
        # resolution), which we bilinearly upsample back to the input size so it overlays
        # smoothly on the image.
        attributions = gradcam.attribute(pixel_values, target=target_output, relu_attributions=True)
        height, width = pixel_values.shape[-2], pixel_values.shape[-1]
        attributions = LayerAttribution.interpolate(attributions, (height, width), interpolate_mode="bilinear")

        display_image = wrapper.get_display_image(pixel_values)
        return self._package_image_output(attributions, display_image, target_output)

    def _package_image_output(self, attributions: torch.Tensor, image: Image.Image, target: int) -> AttributionOutput:
        """Helper to convert raw pixel attributions into the standardized image format."""
        heatmap_payload = render_image_heatmap(attributions, image)
        feature = InputFeature(index=0, content="image", modality="image")
        return AttributionOutput(
            heatmap=[heatmap_payload],
            target=target,
            input_features=[feature],
            metadata={"input_image": image_to_base64(image)},
        )
