import torch
import numpy as np
from typing import Any, List, Optional
from captum.attr import LayerIntegratedGradients

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
# Explicit import for type hinting
from ..wrappers.hf_text import HFTextWrapper

class CaptumGradientsAttributor(BaseAttributor):
    """
    Attributor implementation using Captum's Layer Integrated Gradients.
    Designed to work with HFTextWrapper to explain text model predictions.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None) -> AttributionOutput:
        """
        Computes attribution scores for the given input text.

        Args:
            input_data (str): The raw text to explain.
            target_output (int, optional): The class index to explain. 
                                           If None, the predicted class is used.

        Returns:
            AttributionOutput: The standardized object containing scores and features.
        """
        # Type casting for IDE support
        wrapper: HFTextWrapper = self.wrapper

        # 1. Prepare Inputs
        inputs_dict = wrapper.tokenizer(
            input_data, 
            return_tensors="pt",
            padding=True, 
            truncation=True
        ).to(wrapper.device)
        
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # 2. Define Forward Function for Captum
        # Captum requires a function that maps embeddings to logits.
        # Suggestion: This closure captures 'attention_mask' from the local scope.
        # Moving it to a separate method would require passing the mask explicitly.
        def model_forward(inputs_embeds, mask):
            outputs = wrapper.model(inputs_embeds=inputs_embeds, attention_mask=mask)
            return outputs.logits

        # 3. Initialize Algorithm
        embedding_layer = wrapper.get_embedding_layer()
        lig = LayerIntegratedGradients(model_forward, embedding_layer)

        # 4. Determine Target Class
        if target_output is None:
            logits = wrapper.model(input_ids, attention_mask).logits
            target_output = torch.argmax(logits, dim=1).item()

        # 5. Compute Attributions
        # The attribute method returns the integral of gradients.
        attributions = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=target_output
        )

        # 6. Process Results
        # Suggestion: Modularized normalization logic into a helper method
        normalized_scores = self._process_and_normalize(attributions)

        # 7. Construct Output
        tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids[0])
        features = [
            InputFeature(index=i, content=t, modality="text") 
            for i, t in enumerate(tokens)
        ]

        return AttributionOutput(
            heatmap=normalized_scores,
            target=target_output,
            input_features=features
        )

    def _process_and_normalize(self, attributions: torch.Tensor) -> np.ndarray:
        """
        Aggregates attribution scores across the embedding dimension and normalizes them.

        Args:
            attributions (torch.Tensor): Raw tensor from Captum [1, SeqLen, EmbedDim].

        Returns:
            np.ndarray: Normalized 1D array of scores [SeqLen].
        """
        # Sum across the embedding dimension to get one score per token
        token_scores = attributions.sum(dim=-1).squeeze(0)
        
        # L2 Normalization (Euclidean norm)
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
            
        return token_scores.detach().cpu().numpy()