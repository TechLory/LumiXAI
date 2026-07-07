import torch
from typing import Optional, List, Dict, Any
from captum.attr import DeepLift

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper


class _ClassificationForwardModule(torch.nn.Module):
    """Thin nn.Module adapter around the HF model's forward pass.

    DeepLift (unlike Integrated Gradients/Saliency/etc.) requires its ``forward_func`` to
    be an actual ``nn.Module``: it walks ``.modules()`` to register hooks that override the
    backward pass of non-linearities (its core mechanism), which a plain closure can't
    support. Registering ``hf_model`` as a submodule here makes those hooks reach the real
    model's layers.
    """

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor: # type: ignore
        inputs_embeds = inputs_embeds.contiguous()
        return self.hf_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


class _GenerationForwardModule(torch.nn.Module):
    """Same nn.Module adapter as :class:`_ClassificationForwardModule`, for the
    autoregressive next-token-logit forward pass used by the generative loop."""

    def __init__(self, hf_model: torch.nn.Module, device: str):
        super().__init__()
        self.hf_model = hf_model
        self.device = device

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds = inputs_embeds.contiguous()
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        outputs = self.hf_model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        return outputs.logits[:, -1, :]


class CaptumDeepLiftAttributor(BaseAttributor):
    """Universal Attributor utilizing Captum DeepLift.

    DeepLift is a lighter-weight alternative to Integrated Gradients: it computes
    attributions with a single backward pass against a reference baseline, instead of
    integrating gradients over many interpolation steps. This makes it a better fit for
    lightweight machines while still producing axiomatically-grounded attributions.

    Supports both text classification and autoregressive text generation models, acting
    as a dynamic dispatcher depending on the injected Wrapper type.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Dispatches the attribution request to the appropriate internal method based on the model type.

        Args:
            input_data (str): The raw input text prompt.
            target_output (Optional[int], optional): The target class index (for classification only). Defaults to None.
            **kwargs: Additional parameters for the attribution process.

        Returns:
            AttributionOutput: The structured attribution results.
        """
        if isinstance(self.wrapper, HFTextGenerationWrapper):
            disable_thinking = bool(kwargs.get("disable_thinking", False))
            max_new_tokens = kwargs.get("max_new_tokens", None)
            return self._attribute_generative(input_data, disable_thinking, max_new_tokens)
        else:
            return self._attribute_classification(input_data, target_output)

    # =========================================================
    # 1. CLASSIFICATION (DeepLift)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int]) -> AttributionOutput:
        """Performs DeepLift attribution for sequence classification models.

        Args:
            input_data (str): The input text to classify.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.

        Returns:
            AttributionOutput: A 1D heatmap mapping input tokens to their importance scores.
        """
        wrapper = self.wrapper
        inputs = wrapper.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]

        embeddings = wrapper.get_embedding_layer()(inputs["input_ids"])

        deeplift = DeepLift(_ClassificationForwardModule(wrapper.model))

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        baselines = torch.zeros_like(embeddings)

        attributions = deeplift.attribute(
            inputs=embeddings,
            baselines=baselines,
            target=target_output,
            additional_forward_args=(inputs["attention_mask"],),
        )

        return self._package_output(attributions, inputs["input_ids"][0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive DeepLift)
    # =========================================================
    def _attribute_generative(self, prompt: str, disable_thinking: bool = False, max_new_tokens: Optional[int] = None) -> AttributionOutput:
        """Performs step-by-step DeepLift for autoregressive text generation.

        Mirrors the Integrated Gradients generative loop, but replaces the multi-step
        integration with a single backward pass against a zero-embedding baseline at
        each generated step.

        Args:
            prompt (str): The initial user prompt.

        Returns:
            AttributionOutput: A complex heatmap containing an array of attribution traces for each generated step.
        """
        wrapper = self.wrapper
        print(f"Captum DeepLift: Analyzing '{prompt}' on {wrapper.device}")

        full_text, gen_token_ids, gen_token_strs, gen_probs = wrapper.generate_text(prompt, max_new_tokens=max_new_tokens, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]

        inputs = wrapper.tokenize_generation_prompt(prompt, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]
        current_input_ids = inputs["input_ids"]
        attribution_trace = []

        deeplift = DeepLift(_GenerationForwardModule(wrapper.model, wrapper.device))

        for i, token_str in enumerate(gen_token_strs):
            target_token_id = gen_token_ids[i]
            current_embeddings = wrapper.get_embedding_layer()(current_input_ids)
            baselines = torch.zeros_like(current_embeddings)

            attributions = deeplift.attribute(
                inputs=current_embeddings,
                baselines=baselines,
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
