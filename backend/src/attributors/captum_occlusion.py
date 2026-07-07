import torch
from typing import Optional, List, Dict, Any
from PIL import Image
from captum.attr import Occlusion

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper
from ..wrappers.hf_image_classification import HFImageClassificationWrapper
from ..utils.image_attribution import render_image_heatmap, image_to_base64, decode_base64_image

# Default sliding-window side length (in pixels) used for image classification. Chosen
# to roughly match a ViT-Base patch (16px) while staying independent of any specific
# model's actual patch size, since Occlusion is a black-box method with no notion of
# patches on its own.
DEFAULT_IMAGE_PATCH_SIZE = 16

class CaptumOcclusionAttributor(BaseAttributor):
    """Universal Attributor utilizing Captum Occlusion.

    Occlusion requires no gradients at all: it slides a mask over the input, zeroing one
    whole token embedding (or, for images, one pixel patch) at a time, and measures the
    resulting change in model output. This makes it fully model-agnostic in principle
    (any forward-callable model works), at the cost of one forward pass per
    token/patch rather than a single backward pass — still cheap for the short prompts
    and small images this framework targets.

    Supports text classification, autoregressive text generation, and image classification
    models, acting as a dynamic dispatcher depending on the injected Wrapper type.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Dispatches the attribution request to the appropriate internal method based on the model type.

        Args:
            input_data (str): The raw input text prompt, or (for image classification) a
                PIL Image / base64-encoded image string.
            target_output (Optional[int], optional): The target class index (for classification only). Defaults to None.
            **kwargs: Additional parameters for the attribution process. Accepts a
                `patch_size` override for image classification.

        Returns:
            AttributionOutput: The structured attribution results.
        """
        if isinstance(self.wrapper, HFTextGenerationWrapper):
            disable_thinking = bool(kwargs.get("disable_thinking", False))
            max_new_tokens = kwargs.get("max_new_tokens", None)
            return self._attribute_generative(input_data, disable_thinking, max_new_tokens)
        elif isinstance(self.wrapper, HFImageClassificationWrapper):
            patch_size = kwargs.get("patch_size", DEFAULT_IMAGE_PATCH_SIZE) or DEFAULT_IMAGE_PATCH_SIZE
            return self._attribute_image_classification(input_data, target_output, patch_size)
        else:
            return self._attribute_classification(input_data, target_output)

    # =========================================================
    # 0. IMAGE CLASSIFICATION (Patch-space Occlusion)
    # =========================================================
    def _attribute_image_classification(self, input_data: Any, target_output: Optional[int], patch_size: int) -> AttributionOutput:
        """Performs Occlusion attribution over square pixel patches.

        Args:
            input_data (Any): A PIL Image or a base64-encoded image string.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.
            patch_size (int): Side length (in pixels) of the sliding occlusion window.

        Returns:
            AttributionOutput: A single-entry pixel heatmap mapping image regions to their importance scores.
        """
        wrapper = self.wrapper
        image = input_data if isinstance(input_data, Image.Image) else decode_base64_image(input_data)
        pixel_values = wrapper.preprocess(image)

        def model_forward(pixels):
            return wrapper.model(pixel_values=pixels).logits

        occlusion = Occlusion(model_forward)

        if target_output is None:
            with torch.no_grad():
                logits = wrapper.model(pixel_values=pixel_values).logits
            target_output = torch.argmax(logits, dim=1).item()

        channels = pixel_values.shape[1]
        # Overlapping windows (stride ~= half the window), as in Captum's vision tutorial
        # (window 15, stride 8). Overlap yields a smoother, higher-resolution map than the
        # disjoint blocks that stride == window would produce, at the cost of ~4x more
        # forward passes. The channel stride equals `channels` so the window never slides
        # within the channel dim: every window always spans all channels of a spatial patch.
        stride = max(1, patch_size // 2)

        attributions = occlusion.attribute(
            inputs=pixel_values,
            sliding_window_shapes=(channels, patch_size, patch_size),
            strides=(channels, stride, stride),
            baselines=0,
            target=target_output,
        )

        display_image = wrapper.get_display_image(pixel_values)
        return self._package_image_output(attributions, display_image, target_output)

    # =========================================================
    # 1. CLASSIFICATION (Occlusion)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int]) -> AttributionOutput:
        """Performs Occlusion attribution for sequence classification models.

        Args:
            input_data (str): The input text to classify.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.

        Returns:
            AttributionOutput: A 1D heatmap mapping input tokens to their importance scores.
        """
        wrapper = self.wrapper
        inputs = wrapper.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]

        embeddings = wrapper.get_embedding_layer()(inputs["input_ids"])
        embed_dim = embeddings.shape[-1]

        def model_forward(inputs_embeds, mask):
            inputs_embeds = inputs_embeds.contiguous()
            return wrapper.model(inputs_embeds=inputs_embeds, attention_mask=mask).logits

        occlusion = Occlusion(model_forward)

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        attributions = occlusion.attribute(
            inputs=embeddings,
            sliding_window_shapes=(1, embed_dim),
            strides=(1, embed_dim),
            baselines=0,
            target=target_output,
            additional_forward_args=(inputs["attention_mask"],),
        )

        return self._package_output(attributions, inputs["input_ids"][0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive Occlusion)
    # =========================================================
    def _attribute_generative(self, prompt: str, disable_thinking: bool = False, max_new_tokens: Optional[int] = None) -> AttributionOutput:
        """Performs step-by-step Occlusion for autoregressive text generation.

        Args:
            prompt (str): The initial user prompt.

        Returns:
            AttributionOutput: A complex heatmap containing an array of attribution traces for each generated step.
        """
        wrapper = self.wrapper
        print(f"Captum Occlusion: Analyzing '{prompt}' on {wrapper.device}")

        full_text, gen_token_ids, gen_token_strs, gen_probs = wrapper.generate_text(prompt, max_new_tokens=max_new_tokens, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]

        inputs = wrapper.tokenize_generation_prompt(prompt, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]
        current_input_ids = inputs["input_ids"]
        attribution_trace = []

        def forward_func_adapter(inputs_embeds):
            inputs_embeds = inputs_embeds.contiguous()
            batch_size, seq_len, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=wrapper.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            outputs = wrapper.model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            return outputs.logits[:, -1, :]

        occlusion = Occlusion(forward_func_adapter)

        for i, token_str in enumerate(gen_token_strs):
            target_token_id = gen_token_ids[i]
            current_embeddings = wrapper.get_embedding_layer()(current_input_ids)
            embed_dim = current_embeddings.shape[-1]

            attributions = occlusion.attribute(
                inputs=current_embeddings,
                sliding_window_shapes=(1, embed_dim),
                strides=(1, embed_dim),
                baselines=0,
                target=target_token_id,
            )

            scores = self._normalize(attributions)
            context_tokens = wrapper.tokenizer.convert_ids_to_tokens(current_input_ids[0]) # pyright: ignore[reportAttributeAccessIssue]

            attribution_trace.append({
                "generated_token": token_str,
                "probability": gen_probs[i],
                "context_tokens": context_tokens,
                "attribution_scores": scores
            })

            next_token_tensor = torch.tensor([[target_token_id]]).to(wrapper.device)
            current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=1)

            if wrapper.device.startswith("cuda:"):
                with torch.cuda.device(wrapper.device):
                    torch.cuda.empty_cache()
            elif wrapper.device == "cuda":
                torch.cuda.empty_cache()

        input_special_mask = self.wrapper.get_special_tokens_mask(inputs["input_ids"][0])
        output_special_mask = self.wrapper.get_special_tokens_mask(gen_token_ids)
        input_template_mask = self.wrapper.get_template_tokens_mask(prompt, inputs["input_ids"][0])

        return AttributionOutput(
            heatmap=attribution_trace,
            target="text_generation",
            input_features=[],
            generated_image=None,
            metadata={
                "input_special_mask": input_special_mask,
                "output_special_mask": output_special_mask,
                "input_template_mask": input_template_mask,
            }
        )

    # --- Utilities ---

    def _package_output(self, attributions: torch.Tensor, input_ids: torch.Tensor, target: int) -> AttributionOutput:
        """Helper to convert raw attributions and input IDs into the standardized format."""
        normalized = self._normalize(attributions)
        tokens = self.wrapper.tokenizer.convert_ids_to_tokens(input_ids) # pyright: ignore[reportAttributeAccessIssue]
        features = [InputFeature(index=i, content=t, modality="text") for i, t in enumerate(tokens)]
        special_tokens_mask = self.wrapper.get_special_tokens_mask(input_ids)
        return AttributionOutput(
            heatmap=normalized,
            target=target,
            input_features=features,
            metadata={"special_tokens_mask": special_tokens_mask},
        )

    def _normalize(self, attributions: torch.Tensor) -> List[float]:
        """Normalizes attribution scores across the embedding dimension."""
        token_scores = attributions.sum(dim=-1).squeeze(0)
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
        return token_scores.detach().cpu().numpy().tolist()

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
