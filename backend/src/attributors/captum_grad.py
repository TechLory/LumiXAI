import torch
import numpy as np
from typing import Optional, Union, cast
from captum.attr import LayerIntegratedGradients

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature

from ..wrappers.hf_text_classification import HFTextClassificationWrapper
from ..wrappers.hf_text_generation import HFTextGenerationWrapper

class CaptumGradientsAttributor(BaseAttributor):
    """
    Universal Attributor that uses Captum's Layer Integrated Gradients.
    Supports both text generation models (e.g., GPT, Llama) and text classification models (e.g., BERT).
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None) -> AttributionOutput:
        
        # 1. Recognize the type of Wrapper
        is_generative = isinstance(self.wrapper, HFTextGenerationWrapper)
        is_classification = isinstance(self.wrapper, HFTextClassificationWrapper)

        if not (is_generative or is_classification):
             raise TypeError(f"Unsupported wrapper type: {type(self.wrapper)}")
        
        # 2. Tokenization
        wrapper = self.wrapper
        
        inputs_dict = wrapper.tokenizer( # type: ignore
            input_data, 
            return_tensors="pt",
            padding=True, 
            truncation=True
        ).to(wrapper.device)
        
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # 3. Adaptive Forward Function
        def model_forward(inputs: torch.Tensor, mask: torch.Tensor):

            outputs = wrapper.model(input_ids=inputs, attention_mask=mask)

            if is_generative:
                # (only last token logits)
                return outputs.logits[:, -1, :]
            else:
                # (all logits)
                return outputs.logits

        # 4. Initialize Layer Integrated Gradients
        lig = LayerIntegratedGradients(model_forward, wrapper.get_embedding_layer())

        # 5. Determine Target Output
        if target_output is None:
            logits = wrapper.generate(inputs_dict)
            target_output = torch.argmax(logits, dim=1).item() # pyright: ignore[reportAssignmentType, reportArgumentType]
            
            # Debug
            if is_generative:
                decoded = wrapper.tokenizer.decode([target_output]) # pyright: ignore[reportAttributeAccessIssue]
                print(f"Next Token Predicted: '{decoded}' (ID: {target_output})")

        # 6. Calculate Attributions
        attributions = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=target_output
        )

        # 7. Output Standard
        normalized_scores = self._process_and_normalize(attributions) # pyright: ignore[reportArgumentType]
        tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids[0]) # pyright: ignore[reportAttributeAccessIssue]
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
        token_scores = attributions.sum(dim=-1).squeeze(0)
        norm = torch.norm(token_scores)
        if norm > 0:
            token_scores = token_scores / norm
        return token_scores.detach().cpu().numpy()