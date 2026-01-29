import base64
from io import BytesIO
from typing import Optional, List
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Import custom trace
from ..utils.daam_custom import trace 

from ..abstract import BaseAttributor
from ..schema import AttributionOutput, InputFeature
from ..wrappers.hf_image import HFImageWrapper

class DAAMAttributor(BaseAttributor):
    """
    Attributor for Stable Diffusion using custom DAAM.
    Returns per-token heatmaps.
    """

    def attribute(self, input_data: str, target_output: Optional[int] = None) -> AttributionOutput:
        
        if not isinstance(self.wrapper, HFImageWrapper):
             raise TypeError(f"DAAM requires HFImageWrapper, got: {type(self.wrapper)}")
        
        pipeline = self.wrapper.model
        tokenizer = self.wrapper.model.tokenizer # Retrieve the tokenizer
        prompt = input_data
        
        print(f"Running DAAM (Per-Token) for: '{prompt}'")

        # 1. Tracing
        with trace(pipeline) as tc:
            output = pipeline(prompt, num_inference_steps=20)
            generated_image = output.images[0]
            # Call the function that returns the dictionary {idx: map}
            token_heatmaps = tc.compute_heat_maps()

        # 2. Tokenizing to understand the words
        # Tokenize the prompt to have the ID -> Word correspondence
        tokens = tokenizer.encode(prompt) # Returns list of IDs [49406, 320, ...]
        decoded_tokens = [tokenizer.decode([t]) for t in tokens] 
        # decoded_tokens: ['<|startoftext|>', 'a', 'cat', ..., '<|endoftext|>']

        # 3. Original Image Processing
        buffered_orig = BytesIO()
        generated_image.save(buffered_orig, format="PNG")
        img_str_orig = base64.b64encode(buffered_orig.getvalue()).decode("utf-8")

        # 4. Heatmap Generation for each Token
        heatmap_images = [] # List of Base64 strings
        feature_tokens = [] # List of InputFeature
        
        # Ingnore special tokens to avoid attention sink!
        IGNORED_TOKENS = ["<|startoftext|>", "<|endoftext|>"]

        # Iterate over the prompt tokens (ignore empty padding beyond the end of the prompt)
        for i, token_id in enumerate(tokens):
            word = decoded_tokens[i]
            
            clean_word = word.replace('</w>', '').strip()
            if clean_word in IGNORED_TOKENS or not clean_word:
                continue
            
            if i in token_heatmaps:
                hm_obj = token_heatmaps[i]
                
                # Generate overlay
                fig = hm_obj.plot_overlay(generated_image)
                
                # Save as base64
                buf = BytesIO()
                fig.savefig(buf, format="PNG", bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                heatmap_images.append(b64_str)
                feature_tokens.append(InputFeature(index=i, content=clean_word, modality="text"))

        # 5. Output
        return AttributionOutput(
            heatmap=heatmap_images, # LIST [img1, img2, ...]
            generated_image=img_str_orig,
            target="image_generation",
            input_features=feature_tokens
        )