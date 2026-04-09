import torch
from typing import Optional, List, Dict, Any
from captum.attr import IntegratedGradients

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper

class CaptumGradientsAttributor(BaseAttributor):
    """Universal Attributor utilizing Captum Integrated Gradients.
    
    This class supports both text classification and autoregressive text generation models. 
    It acts as a dynamic dispatcher: depending on the injected Wrapper type, it adapts 
    the forward function and the attribution process to handle either single-pass 
    classification or step-by-step token generation.
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
            return self._attribute_generative(input_data)
        else:
            return self._attribute_classification(input_data, target_output)

    # =========================================================
    # 1. CLASSIFICATION (Standard IG)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int]) -> AttributionOutput:
        """Performs Integrated Gradients attribution for sequence classification models.

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

        ig = IntegratedGradients(model_forward)

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        attributions = ig.attribute(
            inputs=embeddings, 
            target=target_output,
            additional_forward_args=(inputs["attention_mask"],),
            internal_batch_size=2
        )

        return self._package_output(attributions, inputs["input_ids"][0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive IG)
    # =========================================================
    def _attribute_generative(self, prompt: str) -> AttributionOutput:
        """Performs step-by-step Integrated Gradients for autoregressive text generation.

        This method analyzes the causal effect of the evolving context on each newly 
        generated token. It iteratively updates the input embeddings and computes 
        the attribution for the next token in the sequence.

        Args:
            prompt (str): The initial user prompt.

        Returns:
            AttributionOutput: A complex heatmap containing an array of attribution traces for each generated step.
        """
        wrapper = self.wrapper
        print(f"Captum IG: Analyzing '{prompt}' on {wrapper.device}")

        full_text, gen_token_ids, gen_token_strs, gen_probs = wrapper.generate_text(prompt) # pyright: ignore[reportAttributeAccessIssue]
        
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]
        current_input_ids = inputs.input_ids
        attribution_trace = [] 

        def forward_func_adapter(inputs_embeds):
            inputs_embeds = inputs_embeds.contiguous()
            batch_size, seq_len, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=wrapper.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            outputs = wrapper.model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            return outputs.logits[:, -1, :]

        ig = IntegratedGradients(forward_func_adapter)

        for i, token_str in enumerate(gen_token_strs):
            target_token_id = gen_token_ids[i]
            current_embeddings = wrapper.get_embedding_layer()(current_input_ids)

            attributions = ig.attribute(
                inputs=current_embeddings,
                target=target_token_id,
                n_steps=20,
                internal_batch_size=2
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

        return AttributionOutput(
            heatmap=attribution_trace, 
            target="text_generation",
            input_features=[], 
            generated_image=None
        )

    # --- Utilities ---

    def _package_output(self, attributions: torch.Tensor, input_ids: torch.Tensor, target: int) -> AttributionOutput:
        """Helper to convert raw attributions and input IDs into the standardized format."""
        normalized = self._normalize(attributions)
        tokens = self.wrapper.tokenizer.convert_ids_to_tokens(input_ids) # pyright: ignore[reportAttributeAccessIssue]
        features = [InputFeature(index=i, content=t, modality="text") for i, t in enumerate(tokens)]
        return AttributionOutput(heatmap=normalized, target=target, input_features=features)

    def _normalize(self, attributions: torch.Tensor) -> List[float]:
        """Normalizes attribution scores across the embedding dimension."""
        token_scores = attributions.sum(dim=-1).squeeze(0)
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
        return token_scores.detach().cpu().numpy().tolist()
