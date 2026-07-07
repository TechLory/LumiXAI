import torch
from typing import Optional, List, Dict, Any
from PIL import Image
from captum.attr import Saliency

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper
from ..wrappers.hf_image_classification import HFImageClassificationWrapper
from ..utils.image_attribution import render_image_heatmap, image_to_base64, decode_base64_image

class CaptumSaliencyAttributor(BaseAttributor):
    """Universal Attributor utilizing Captum Saliency.

    Saliency is the cheapest attribution method available: a single gradient-of-output
    w.r.t.-input pass, with no baseline and no integration steps. It is a coarser signal
    than Integrated Gradients or DeepLift, but well suited to lightweight machines.

    Supports text classification, autoregressive text generation, and image classification
    models, acting as a dynamic dispatcher depending on the injected Wrapper type.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Dispatches the attribution request to the appropriate internal method based on the model type.

        Args:
            input_data (str): The raw input text prompt, or (for image classification) a
                PIL Image / base64-encoded image string.
            target_output (Optional[int], optional): The target class index (for classification only). Defaults to None.
            **kwargs: Additional parameters for the attribution process.

        Returns:
            AttributionOutput: The structured attribution results.
        """
        if isinstance(self.wrapper, HFTextGenerationWrapper):
            disable_thinking = bool(kwargs.get("disable_thinking", False))
            max_new_tokens = kwargs.get("max_new_tokens", None)
            return self._attribute_generative(input_data, disable_thinking, max_new_tokens)
        elif isinstance(self.wrapper, HFImageClassificationWrapper):
            return self._attribute_image_classification(input_data, target_output)
        else:
            return self._attribute_classification(input_data, target_output)

    # =========================================================
    # 0. IMAGE CLASSIFICATION (Pixel-space Saliency)
    # =========================================================
    def _attribute_image_classification(self, input_data: Any, target_output: Optional[int]) -> AttributionOutput:
        """Performs Saliency attribution directly on pixel values.

        Args:
            input_data (Any): A PIL Image or a base64-encoded image string.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.

        Returns:
            AttributionOutput: A single-entry pixel heatmap mapping image regions to their importance scores.
        """
        wrapper = self.wrapper
        image = input_data if isinstance(input_data, Image.Image) else decode_base64_image(input_data)
        pixel_values = wrapper.preprocess(image)

        def model_forward(pixels):
            return wrapper.model(pixel_values=pixels).logits

        saliency = Saliency(model_forward)

        if target_output is None:
            with torch.no_grad():
                logits = wrapper.model(pixel_values=pixel_values).logits
            target_output = torch.argmax(logits, dim=1).item()

        attributions = saliency.attribute(inputs=pixel_values, target=target_output)

        return self._package_image_output(attributions, image, target_output)

    # =========================================================
    # 1. CLASSIFICATION (Saliency)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int]) -> AttributionOutput:
        """Performs Saliency attribution for sequence classification models.

        Args:
            input_data (str): The input text to classify.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.

        Returns:
            AttributionOutput: A 1D heatmap mapping input tokens to their importance scores.
        """
        wrapper = self.wrapper
        inputs = wrapper.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]

        embeddings = wrapper.get_embedding_layer()(inputs["input_ids"])

        def model_forward(inputs_embeds, mask):
            inputs_embeds = inputs_embeds.contiguous()
            return wrapper.model(inputs_embeds=inputs_embeds, attention_mask=mask).logits

        saliency = Saliency(model_forward)

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        attributions = saliency.attribute(
            inputs=embeddings,
            target=target_output,
            additional_forward_args=(inputs["attention_mask"],),
        )

        return self._package_output(attributions, inputs["input_ids"][0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive Saliency)
    # =========================================================
    def _attribute_generative(self, prompt: str, disable_thinking: bool = False, max_new_tokens: Optional[int] = None) -> AttributionOutput:
        """Performs step-by-step Saliency for autoregressive text generation.

        Args:
            prompt (str): The initial user prompt.

        Returns:
            AttributionOutput: A complex heatmap containing an array of attribution traces for each generated step.
        """
        wrapper = self.wrapper
        print(f"Captum Saliency: Analyzing '{prompt}' on {wrapper.device}")

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

        saliency = Saliency(forward_func_adapter)

        for i, token_str in enumerate(gen_token_strs):
            target_token_id = gen_token_ids[i]
            current_embeddings = wrapper.get_embedding_layer()(current_input_ids)

            attributions = saliency.attribute(
                inputs=current_embeddings,
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
