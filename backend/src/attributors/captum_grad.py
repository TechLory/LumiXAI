import torch
import numpy as np
from typing import Any, List, Optional
from captum.attr import LayerIntegratedGradients

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_text import HFTextWrapper

class CaptumGradientsAttributor(BaseAttributor):
    """
    Attributor implementation using Captum's Layer Integrated Gradients.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None) -> AttributionOutput:
        
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
        def model_forward(inputs, mask):
            return wrapper.model(input_ids=inputs, attention_mask=mask).logits

        # 3. Initialize Algorithm
        embedding_layer = wrapper.get_embedding_layer()
        lig = LayerIntegratedGradients(model_forward, embedding_layer)

        # 4. Determine Target Class
        if target_output is None:
            # Calculate logits to find the predicted class
            logits = wrapper.model(input_ids, attention_mask).logits
            target_output = torch.argmax(logits, dim=1).item()

        # 5. Compute Attributions
        attributions = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=target_output
        )

        # 6. Process Results
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
        # Sum across the embedding dimension to get one score per token
        token_scores = attributions.sum(dim=-1).squeeze(0)
        
        # L2 Normalization
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
            
        return token_scores.detach().cpu().numpy()