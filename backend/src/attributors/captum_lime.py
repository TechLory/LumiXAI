import torch
from typing import Optional, List, Dict, Any
from captum.attr import Lime
from captum._utils.models.linear_model import SGDLinearRegression

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper

# LIME's cost is dominated by `n_samples` (one forward pass per sample), not input size,
# so this is the main lever for keeping it usable on lightweight machines. Generation uses
# a lower default since the cost multiplies by the number of generated tokens.
DEFAULT_N_SAMPLES_CLASSIFICATION = 25
DEFAULT_N_SAMPLES_GENERATION = 15


class _DeviceAwareSGDLinearRegression(SGDLinearRegression):
    """Captum's `Lime.attribute()` never forwards a `device` kwarg to the
    surrogate's `.fit()` call, so `SGDLinearRegression`'s underlying
    `nn.Linear` is always lazily built on the CPU (see
    `LinearModel._construct_model_params`), regardless of where the
    perturbation samples it's trained on live. On a GPU-hosted model those
    samples are on `cuda`, so the first forward pass through the freshly
    built CPU layer crashes with a device mismatch. Moving the module to the
    target device right after it's constructed fixes this without needing to
    fork Captum's training loop.
    """

    def __init__(self, device: str, **kwargs):
        super().__init__(**kwargs)
        self._target_device = device

    def _construct_model_params(self, *args, **kwargs):
        super()._construct_model_params(*args, **kwargs)
        self.to(self._target_device)


class CaptumLimeAttributor(BaseAttributor):
    """Universal Attributor utilizing Captum LIME.

    Unlike the gradient-based attributors in this package, LIME is fully model-agnostic:
    it perturbs the input at the token level (replacing tokens with a baseline id
    according to a sampled binary mask) and fits a local linear surrogate model to the
    resulting logit changes. No gradients or embeddings are required for the perturbation
    itself, only a forward pass per sample.

    Supports both text classification and autoregressive text generation models, acting
    as a dynamic dispatcher depending on the injected Wrapper type.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Dispatches the attribution request to the appropriate internal method based on the model type.

        Args:
            input_data (str): The raw input text prompt.
            target_output (Optional[int], optional): The target class index (for classification only). Defaults to None.
            **kwargs: Additional parameters. Accepts an `n_samples` override.

        Returns:
            AttributionOutput: The structured attribution results.
        """
        if isinstance(self.wrapper, HFTextGenerationWrapper):
            n_samples = kwargs.get("n_samples", DEFAULT_N_SAMPLES_GENERATION) or DEFAULT_N_SAMPLES_GENERATION
            disable_thinking = bool(kwargs.get("disable_thinking", False))
            max_new_tokens = kwargs.get("max_new_tokens", None)
            return self._attribute_generative(input_data, n_samples, disable_thinking, max_new_tokens)
        else:
            n_samples = kwargs.get("n_samples", DEFAULT_N_SAMPLES_CLASSIFICATION) or DEFAULT_N_SAMPLES_CLASSIFICATION
            return self._attribute_classification(input_data, target_output, n_samples)

    def _get_baseline_token_id(self) -> int:
        """Picks a reasonable "absent" token id to substitute for occluded features."""
        tokenizer = self.wrapper.tokenizer # pyright: ignore[reportAttributeAccessIssue]
        baseline_id = getattr(tokenizer, "pad_token_id", None)
        if baseline_id is None:
            baseline_id = getattr(tokenizer, "unk_token_id", None)
        if baseline_id is None:
            baseline_id = 0
        return baseline_id

    # =========================================================
    # 1. CLASSIFICATION (LIME)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int], n_samples: int) -> AttributionOutput:
        """Performs LIME attribution for sequence classification models.

        Args:
            input_data (str): The input text to classify.
            target_output (Optional[int]): The specific class to attribute towards. If None, the predicted class is used.
            n_samples (int): Number of perturbed samples used to fit the local surrogate model.

        Returns:
            AttributionOutput: A 1D heatmap mapping input tokens to their importance scores.
        """
        wrapper = self.wrapper
        inputs = wrapper.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_len = input_ids.shape[1]

        def forward_func(perturbed_input_ids, mask):
            embeddings = wrapper.get_embedding_layer()(perturbed_input_ids)
            return wrapper.model(inputs_embeds=embeddings, attention_mask=mask).logits

        # Use captum's pure-PyTorch SGD linear surrogate instead of the sklearn-backed
        # default, avoiding a new heavyweight dependency for this one method.
        lime = Lime(forward_func, interpretable_model=_DeviceAwareSGDLinearRegression(wrapper.device))

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        baseline_token_id = self._get_baseline_token_id()
        baselines = torch.full_like(input_ids, baseline_token_id)
        feature_mask = torch.arange(seq_len, device=wrapper.device).unsqueeze(0)

        attributions = lime.attribute(
            inputs=input_ids,
            baselines=baselines,
            target=target_output,
            additional_forward_args=(attention_mask,),
            feature_mask=feature_mask,
            n_samples=n_samples,
        )

        return self._package_output(attributions, input_ids[0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive LIME)
    # =========================================================
    def _attribute_generative(self, prompt: str, n_samples: int, disable_thinking: bool = False, max_new_tokens: Optional[int] = None) -> AttributionOutput:
        """Performs step-by-step LIME for autoregressive text generation.

        Args:
            prompt (str): The initial user prompt.
            n_samples (int): Number of perturbed samples used to fit the local surrogate model, per step.

        Returns:
            AttributionOutput: A complex heatmap containing an array of attribution traces for each generated step.
        """
        wrapper = self.wrapper
        print(f"Captum LIME: Analyzing '{prompt}' on {wrapper.device}")

        full_text, gen_token_ids, gen_token_strs, gen_probs = wrapper.generate_text(prompt, max_new_tokens=max_new_tokens, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]

        inputs = wrapper.tokenize_generation_prompt(prompt, disable_thinking=disable_thinking) # pyright: ignore[reportAttributeAccessIssue]
        current_input_ids = inputs["input_ids"]
        attribution_trace = []
        baseline_token_id = self._get_baseline_token_id()

        def forward_func_adapter(perturbed_input_ids):
            embeddings = wrapper.get_embedding_layer()(perturbed_input_ids)
            batch_size, seq_len, _ = embeddings.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=wrapper.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            outputs = wrapper.model(inputs_embeds=embeddings, position_ids=position_ids)
            return outputs.logits[:, -1, :]

        lime = Lime(forward_func_adapter, interpretable_model=_DeviceAwareSGDLinearRegression(wrapper.device))

        for i, token_str in enumerate(gen_token_strs):
            target_token_id = gen_token_ids[i]
            seq_len = current_input_ids.shape[1]
            baselines = torch.full_like(current_input_ids, baseline_token_id)
            feature_mask = torch.arange(seq_len, device=wrapper.device).unsqueeze(0)

            attributions = lime.attribute(
                inputs=current_input_ids,
                baselines=baselines,
                target=target_token_id,
                feature_mask=feature_mask,
                n_samples=n_samples,
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
        """Normalizes per-token attribution scores.

        Unlike the embedding-based attributors, LIME's output already has one scalar per
        token position (shape ``[1, seq_len]``) since it perturbs discrete token ids
        directly rather than a continuous embedding tensor — no summation over an
        embedding dimension is needed here.
        """
        token_scores = attributions.squeeze(0).float()
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
        return token_scores.detach().cpu().numpy().tolist()
