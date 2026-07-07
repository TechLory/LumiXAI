import torch
from typing import Optional, List, Dict, Any
from captum.attr import GradientShap

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper

# Small, fixed sample counts keep GradientSHAP cheap on lightweight (CPU-only) machines:
# each sample is one noisy forward+backward pass, so cost scales linearly with this value.
DEFAULT_N_SAMPLES = 5
DEFAULT_STDEVS = 0.01

class CaptumGradientShapAttributor(BaseAttributor):
    """Universal Attributor utilizing Captum GradientSHAP.

    GradientSHAP approximates Shapley values by averaging gradients over a handful of
    noisy interpolations between the input and a baseline distribution. It is more
    expensive than Saliency/DeepLift but much cheaper than sampling-based SHAP, since the
    number of samples (not the input size) controls the cost.

    Supports both text classification and autoregressive text generation models, acting
    as a dynamic dispatcher depending on the injected Wrapper type.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Dispatches the attribution request to the appropriate internal method based on the model type.

        Args:
            input_data (str): The raw input text prompt.
            target_output (Optional[int], optional): The target class index (for classification only). Defaults to None.
            **kwargs: Additional parameters. Accepts `n_samples` and `stdevs` overrides.

        Returns:
            AttributionOutput: The structured attribution results.
        """
        n_samples = kwargs.get("n_samples", DEFAULT_N_SAMPLES) or DEFAULT_N_SAMPLES
        stdevs = kwargs.get("stdevs", DEFAULT_STDEVS) or DEFAULT_STDEVS

        if isinstance(self.wrapper, HFTextGenerationWrapper):
            disable_thinking = bool(kwargs.get("disable_thinking", False))
            return self._attribute_generative(input_data, n_samples, stdevs, disable_thinking)
        else:
            return self._attribute_classification(input_data, target_output, n_samples, stdevs)

    # =========================================================
    # 1. CLASSIFICATION (GradientSHAP)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int], n_samples: int, stdevs: float) -> AttributionOutput:
        """Performs GradientSHAP attribution for sequence classification models.

        Args:
            input_data (str): The input text to classify.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.
            n_samples (int): Number of noisy samples used to approximate the expectation.
            stdevs (float): Standard deviation of the Gaussian noise added around each baseline.

        Returns:
            AttributionOutput: A 1D heatmap mapping input tokens to their importance scores.
        """
        wrapper = self.wrapper
        inputs = wrapper.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]

        embeddings = wrapper.get_embedding_layer()(inputs["input_ids"])

        def model_forward(inputs_embeds, mask):
            inputs_embeds = inputs_embeds.contiguous()
            return wrapper.model(inputs_embeds=inputs_embeds, attention_mask=mask).logits

        gradient_shap = GradientShap(model_forward)

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        # Baseline distribution: a couple of zero-embedding references is enough for
        # GradientSHAP's noise-interpolation scheme (it does not need real data baselines).
        baselines = torch.zeros_like(embeddings).repeat(2, 1, 1)

        attributions = gradient_shap.attribute(
            inputs=embeddings,
            baselines=baselines,
            n_samples=n_samples,
            stdevs=stdevs,
            target=target_output,
            additional_forward_args=(inputs["attention_mask"],),
        )

        return self._package_output(attributions, inputs["input_ids"][0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive GradientSHAP)
    # =========================================================
    def _attribute_generative(self, prompt: str, n_samples: int, stdevs: float, disable_thinking: bool = False) -> AttributionOutput:
        """Performs step-by-step GradientSHAP for autoregressive text generation.

        Args:
            prompt (str): The initial user prompt.
            n_samples (int): Number of noisy samples used to approximate the expectation, per step.
            stdevs (float): Standard deviation of the Gaussian noise added around each baseline.

        Returns:
            AttributionOutput: A complex heatmap containing an array of attribution traces for each generated step.
        """
        wrapper = self.wrapper
        print(f"Captum GradientSHAP: Analyzing '{prompt}' on {wrapper.device}")

        full_text, gen_token_ids, gen_token_strs, gen_probs = wrapper.generate_text(prompt, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]

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

        gradient_shap = GradientShap(forward_func_adapter)

        for i, token_str in enumerate(gen_token_strs):
            target_token_id = gen_token_ids[i]
            current_embeddings = wrapper.get_embedding_layer()(current_input_ids)
            baselines = torch.zeros_like(current_embeddings).repeat(2, 1, 1)

            attributions = gradient_shap.attribute(
                inputs=current_embeddings,
                baselines=baselines,
                n_samples=n_samples,
                stdevs=stdevs,
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
