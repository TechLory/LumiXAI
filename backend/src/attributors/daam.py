import base64
from io import BytesIO
from typing import Optional, List
import matplotlib
from prompt_toolkit import prompt
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ..utils.daam_custom import trace 
from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_image import HFImageWrapper

class DAAMAttributor(BaseAttributor):
    """Diffusion Attentive Attribution Maps (DAAM) Attributor for Stable Diffusion models.
    
    This class extracts spatial cross-attention maps during the image generation process,
    linking specific text tokens in the prompt to pixel regions in the generated image.
    It provides both visual overlays (Base64 PNGs) and raw numeric matrices for interactive UIs.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None, **kwargs) -> AttributionOutput:
        """Executes the DAAM tracing and generation pipeline.

        Notes:
            **Extending Model Support:** The inference logic dynamically adjusts `num_inference_steps`, `guidance_scale`, 
            and `negative_prompt` based on the architecture (e.g., standard vs. Turbo models) 
            to prevent math scheduler crashes.
            If a new model crashes or requires specific hyperparameters (e.g., LCM models), 
            developers must add a new `elif` condition checking `model_id_lower` inside the 
            `with trace(pipeline)` block.
            *Example for LCM:* `elif "lcm" in model_id_lower: inference_steps = 4; guidance_scale = 1.5; neg_prompt = None`

        Args:
            input_data (str): The text prompt used to generate the image.
            target_output (Optional[int], optional): Not used for image generation. Defaults to None.
            **kwargs: Additional options. Pass `ignore_special_tokens=True` to exclude BOS/EOS tags from the results.

        Returns:
            AttributionOutput: The structured output where the `heatmap` field contains a list of dictionaries. 
                Each dictionary holds a Base64 string (`image_base64`) and a 64x64 interpolated matrix (`raw_matrix`) for a specific token.

        Raises:
            TypeError: If the injected wrapper is not an instance of HFImageWrapper.
        """
        if not isinstance(self.wrapper, HFImageWrapper):
             raise TypeError(f"DAAM requires HFImageWrapper, got: {type(self.wrapper)}")
        
        pipeline = self.wrapper.model
        tokenizer = self.wrapper.model.tokenizer 
        prompt = input_data
        
        print(f"Running DAAM (Per-Token) for: '{prompt}'")

        # 1. Tracing
        with trace(pipeline) as tc:
            model_id_lower = self.wrapper.model_id.lower()
            
            if "sdxl-turbo" in model_id_lower:
                inference_steps = 4
                guidance_scale = 0.0
                neg_prompt = None
            elif "turbo" in model_id_lower:
                inference_steps = 1
                guidance_scale = 0.0
                neg_prompt = None
            else:
                inference_steps = 30
                guidance_scale = 7.5
                neg_prompt = "blurry, low quality, distortion, ugly, bad anatomy, watermark, text"

            pipeline_args = {
                "prompt": prompt,
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale
            }

            if neg_prompt is not None:
                pipeline_args["negative_prompt"] = neg_prompt

            output = pipeline(**pipeline_args)

            generated_image = output.images[0]
            token_heatmaps = tc.compute_heat_maps()

        # 2. Tokenizing to understand the words
        tokens = tokenizer.encode(prompt) 
        decoded_tokens = [tokenizer.decode([t]) for t in tokens] 

        # 3. Original Image Processing
        buffered_orig = BytesIO()
        generated_image.save(buffered_orig, format="PNG")
        img_str_orig = base64.b64encode(buffered_orig.getvalue()).decode("utf-8")

        # 4. Heatmap Generation for each Token
        heatmap_data = [] 
        feature_tokens = [] 
        
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
                raw_tensor = hm_obj.heatmap                
                raw_tensor = raw_tensor.unsqueeze(0).unsqueeze(0)            
                small_tensor = F.interpolate(raw_tensor.to(torch.float32), size=(64, 64), mode='bilinear', align_corners=False).to(raw_tensor.dtype)
                raw_matrix = small_tensor.squeeze().cpu().tolist()

                heatmap_data.append({
                    "image_base64": b64_str,
                    "raw_matrix": raw_matrix
                })
                
                feature_tokens.append(InputFeature(index=i, content=clean_word, modality="text"))

        # 5. Output
        return AttributionOutput(
            heatmap=heatmap_data, 
            generated_image=img_str_orig,
            target="image_generation",
            input_features=feature_tokens
        )