import torch
from typing import Any, Union, List
from diffusers import StableDiffusionPipeline
from ..abstract import BaseWrapper

class HFImageWrapper(BaseWrapper):
    """
    Wrapper for Text-to-Image Generation using Stable Diffusion.
    Uses Hugging Face 'diffusers' library.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        print(f"Loading Stable Diffusion Pipeline: {self.model_id}...")
        
        # Optimize memory: use float16 for CUDA/MPS (Mac), float32 for CPU
        torch_dtype = torch.float32
        if self.device != "cpu":
            torch_dtype = torch.float16

        try:
            # Load the full pipeline (UNet, VAE, Text Encoder, Scheduler)
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch_dtype,
                #variant="fp16",
                safety_checker=None
            )
            
            # Avoid 
            if self.device == "cuda":
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.to(self.device)
            
            # DEBUG:Disable safety checker for debugging/XAI purposes (avoids black images)
            #if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            #   pipe.safety_checker = None
                
            return pipe
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Stable Diffusion model: {e}")

    def generate(self, input_data: str) -> Any:
        """
        Runs the generation pipeline.
        Input: Prompt string.
        Output: PIL Image object.
        """
        if not isinstance(input_data, str):
            raise ValueError("Input for HFImageWrapper must be a text prompt string.")

        print(f"Generating image for prompt: '{input_data}'...")
        
        # Inference step
        # num_inference_steps defaults to 50, (lowered for speed in testing)
        with torch.no_grad():
            output = self.model(input_data, num_inference_steps=30)
            
        # Returns the first generated image (PIL format)
        return output.images[0]

    def get_embedding_layer(self) -> torch.nn.Module:
        """
        Returns the embedding layer of the Text Encoder (CLIP).
        Required for token-level analysis.
        """
        return self.model.text_encoder.get_input_embeddings()