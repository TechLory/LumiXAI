import torch
from typing import Optional, Any
from PIL import Image
from captum.attr import Saliency, NoiseTunnel

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_image_classification import HFImageClassificationWrapper
from ..utils.image_attribution import render_image_heatmap, image_to_base64, decode_base64_image

# SmoothGrad averages the gradient over `nt_samples` copies of the input, each perturbed
# with Gaussian noise. This is Captum's documented remedy (in the vision tutorials, via
# NoiseTunnel) for the salt-and-pepper look of raw single-pass gradient saliency: the
# per-pixel noise cancels on average while the true signal reinforces. Cost is one
# backward pass per sample, so this is the main runtime lever on CPU-only machines.
DEFAULT_NT_SAMPLES = 15
# Std of the Gaussian noise added to the (normalized) pixel tensor before each gradient.
# Too small and no smoothing happens; too large and the map blurs into the background.
# 0.15 tracks the ~0.2 used in Captum's TorchVision tutorial for normalized inputs.
DEFAULT_STDEVS = 0.15
# Cap how many noisy copies are batched through the model at once, to bound peak memory
# on small machines (the samples are otherwise stacked into a single large batch).
NT_SAMPLES_BATCH_SIZE = 4


class CaptumSmoothGradAttributor(BaseAttributor):
    """Image-classification attributor: SmoothGrad via Captum's NoiseTunnel over Saliency.

    Unlike the other Captum attributors in this package, SmoothGrad is exposed only for
    image classification: it exists specifically to clean up the noisy pixel-space
    gradients that vanilla Saliency produces on images, which is why Captum documents it
    in the vision tutorials. It is registered with `hf_image_classification` as its only
    compatible wrapper, so the universal text dispatch other attributors carry is not
    needed here.
    """

    def attribute(self, input_data: Any, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Computes a SmoothGrad saliency map for an image classification model.

        Args:
            input_data (Any): A PIL Image or a base64-encoded image string.
            target_output (Optional[int], optional): The target class index. If None, the
                predicted class is used.
            **kwargs: Accepts `nt_samples` and `stdevs` overrides.

        Returns:
            AttributionOutput: A single-entry pixel heatmap mapping image regions to importance.
        """
        if not isinstance(self.wrapper, HFImageClassificationWrapper):
            raise NotImplementedError(
                "SmoothGrad (Captum) is only available for image classification models."
            )

        nt_samples = kwargs.get("nt_samples") or DEFAULT_NT_SAMPLES
        stdevs = kwargs.get("stdevs") or DEFAULT_STDEVS

        wrapper = self.wrapper
        image = input_data if isinstance(input_data, Image.Image) else decode_base64_image(input_data)
        pixel_values = wrapper.preprocess(image)

        def model_forward(pixels):
            return wrapper.model(pixel_values=pixels).logits

        smoothgrad = NoiseTunnel(Saliency(model_forward))

        if target_output is None:
            with torch.no_grad():
                logits = wrapper.model(pixel_values=pixel_values).logits
            target_output = torch.argmax(logits, dim=1).item()

        attributions = smoothgrad.attribute(
            inputs=pixel_values,
            target=target_output,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            nt_samples_batch_size=NT_SAMPLES_BATCH_SIZE,
            stdevs=stdevs,
        )

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
