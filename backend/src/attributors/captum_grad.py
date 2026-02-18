import torch
from typing import Optional, List, Dict, Any
from captum.attr import IntegratedGradients

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text_generation import HFTextGenerationWrapper

class CaptumGradientsAttributor(BaseAttributor):
    """
    Universal Attributor that uses Captum Integrated Gradients.
    Supports both classification and autoregressive text generation models by adapting the forward function and attribution process accordingly.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None) -> AttributionOutput:
        
        # --- Dispatcher ---
        if isinstance(self.wrapper, HFTextGenerationWrapper):
            return self._attribute_generative(input_data)
        else:
            return self._attribute_classification(input_data, target_output)

    # =========================================================
    # 1. CLASSIFICATION (Standard IG)
    # =========================================================
    def _attribute_classification(self, input_data: str, target_output: Optional[int]) -> AttributionOutput:
        wrapper = self.wrapper
        inputs = wrapper.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]
        
        # Calculate Embeddings
        embeddings = wrapper.get_embedding_layer()(inputs["input_ids"]) # [1, Seq, Hidden]

        # Embeddings -> Logits
        def model_forward(inputs_embeds):
            inputs_embeds = inputs_embeds.contiguous()
            return wrapper.model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"]).logits

        ig = IntegratedGradients(model_forward)

        if target_output is None:
            logits = wrapper.model(**inputs).logits
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType]

        # Attribution
        attributions = ig.attribute(inputs=embeddings, target=target_output)

        return self._package_output(attributions, inputs["input_ids"][0], target_output)

    # =========================================================
    # 2. GENERATION (Autoregressive IG)
    # =========================================================
    def _attribute_generative(self, prompt: str) -> AttributionOutput:
        wrapper = self.wrapper
        print(f"Captum IG: Analyzing '{prompt}' on {wrapper.device}")

        # Text Generation (gets generated tokens and probabilities)
        full_text, gen_token_strs, gen_probs = wrapper.generate_text(prompt, max_new_tokens=20) # pyright: ignore[reportAttributeAccessIssue]
        
        # Preprocess input prompt (initial context)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device) # pyright: ignore[reportAttributeAccessIssue]
        current_input_ids = inputs.input_ids
        attribution_trace = [] 

        def forward_func_adapter(inputs_embeds):
            # inputs_embeds: [Batch, SeqLen, Hidden]
            inputs_embeds = inputs_embeds.contiguous()
            
            batch_size, seq_len, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=wrapper.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            outputs = wrapper.model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            return outputs.logits[:, -1, :]

        ig = IntegratedGradients(forward_func_adapter)

        # Loop on generated tokens
        for i, token_str in enumerate(gen_token_strs):
            target_token_id = wrapper.tokenizer.encode(token_str, add_special_tokens=False)[0] # pyright: ignore[reportAttributeAccessIssue]

            # Gets current embeddings
            current_embeddings = wrapper.get_embedding_layer()(current_input_ids) # [1, Len, Hidden]

            # Attribution
            attributions = ig.attribute(
                inputs=current_embeddings,
                target=target_token_id,
                n_steps=20 
            )

            # Save attribution scores and context tokens for this step
            scores = self._normalize(attributions)
            context_tokens = wrapper.tokenizer.convert_ids_to_tokens(current_input_ids[0]) # pyright: ignore[reportAttributeAccessIssue]
            
            attribution_trace.append({
                "generated_token": token_str,
                "probability": gen_probs[i],
                "context_tokens": context_tokens,
                "attribution_scores": scores
            })

            # Update input_ids for next step (append generated token)
            next_token_tensor = torch.tensor([[target_token_id]]).to(wrapper.device)
            current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=1)

        return AttributionOutput(
            heatmap=attribution_trace, 
            target="text_generation",
            input_features=[], 
            generated_image=None
        )

    # --- Utilities ---

    def _package_output(self, attributions, input_ids, target):
        """Helper to convert attributions and input_ids into AttributionOutput format"""
        normalized = self._normalize(attributions)
        tokens = self.wrapper.tokenizer.convert_ids_to_tokens(input_ids) # pyright: ignore[reportAttributeAccessIssue]
        features = [InputFeature(index=i, content=t, modality="text") for i, t in enumerate(tokens)]
        return AttributionOutput(heatmap=normalized, target=target, input_features=features)

    def _normalize(self, attributions: torch.Tensor) -> List[float]:
        """Normalize attribution scores by summing over the embedding dimension and then normalizing the resulting token scores."""
        token_scores = attributions.sum(dim=-1).squeeze(0)
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
        return token_scores.detach().cpu().numpy().tolist()