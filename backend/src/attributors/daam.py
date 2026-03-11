import base64
from io import BytesIO
from typing import Optional, List
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Import custom trace
from ..utils.daam_custom import trace 

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_image import HFImageWrapper

class DAAMAttributor(BaseAttributor):
    """
    Attributor for Stable Diffusion using custom DAAM.
    Returns per-token heatmaps (Base64) AND raw 64x64 matrices for interactive UX.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        
        if not isinstance(self.wrapper, HFImageWrapper):
             raise TypeError(f"DAAM requires HFImageWrapper, got: {type(self.wrapper)}")
        
        pipeline = self.wrapper.model
        tokenizer = self.wrapper.model.tokenizer # Retrieve the tokenizer
        prompt = input_data
        
        print(f"Running DAAM (Per-Token) for: '{prompt}'")

        # 1. Tracing
        with trace(pipeline) as tc:

            # Negative Prompt
            neg_prompt = "blurry, low quality, distortion, ugly, bad anatomy, watermark, text"

            output = pipeline(
                prompt, 
                negative_prompt=neg_prompt,
                num_inference_steps=30
            )
            generated_image = output.images[0]
            # Call the function that returns the dictionary {idx: map}
            token_heatmaps = tc.compute_heat_maps()

        # 2. Tokenizing to understand the words
        tokens = tokenizer.encode(prompt) # Returns list of IDs [49406, 320, ...]
        decoded_tokens = [tokenizer.decode([t]) for t in tokens] 

        # 3. Original Image Processing
        buffered_orig = BytesIO()
        generated_image.save(buffered_orig, format="PNG")
        img_str_orig = base64.b64encode(buffered_orig.getvalue()).decode("utf-8")

        # 4. Heatmap Generation for each Token
        heatmap_data = [] # List of dictionaries: [{"image_base64": ..., "raw_matrix": [...]}, ...]
        feature_tokens = [] 
        
        # FILTER SPECIAL TOKENS SETUP (using tokenizer's special IDs)
        ignore_special_tokens = kwargs.get("ignore_special_tokens", True)
        special_ids = set()
        if getattr(tokenizer, "all_special_ids", None):
            special_ids.update(tokenizer.all_special_ids)
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            val = getattr(tokenizer, attr, None)
            if val is not None:
                special_ids.add(val)
        special_ids.update([49406, 49407])
        special_strings = ["<|startoftext|>", "<|endoftext|>"]
        
        for i, token_id in enumerate(tokens):
            word = decoded_tokens[i]
            clean_word = word.replace('</w>', '').strip()

            # Filtering
            is_special = (token_id in special_ids) or (clean_word in special_strings)
            if not clean_word or (ignore_special_tokens and is_special):
                continue
            
            if i in token_heatmaps:
                hm_obj = token_heatmaps[i]
                
                # --- A. Generation of Base64 Image ---
                fig = hm_obj.plot_overlay(generated_image)
                buf = BytesIO()
                fig.savefig(buf, format="PNG", bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                # --- B. Raw Data Extraction ---
                raw_tensor = hm_obj.heatmap # PyTorch Tensor (eg. 512x512)                
                raw_tensor = raw_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]                
                small_tensor = F.interpolate(raw_tensor.to(torch.float32), size=(64, 64), mode='bilinear', align_corners=False).to(raw_tensor.dtype)
                raw_matrix = small_tensor.squeeze().cpu().tolist()

                heatmap_data.append({
                    "image_base64": b64_str,
                    "raw_matrix": raw_matrix
                })
                
                feature_tokens.append(InputFeature(index=i, content=clean_word, modality="text"))

        # 5. Output
        return AttributionOutput(
            heatmap=heatmap_data, # LIST OF DICTIONARIES: [{"image_base64": ..., "raw_matrix": [...]}, ...]
            generated_image=img_str_orig,
            target="image_generation",
            input_features=feature_tokens
        )