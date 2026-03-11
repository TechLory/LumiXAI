import torch
from typing import Any, Union, List
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from ..abstract import BaseWrapper

class HFImageWrapper(BaseWrapper):
    """
    Wrapper for Text-to-Image Generation using Stable Diffusion.
    Uses Hugging Face 'diffusers' library.
    Supports both Stable Diffusion and Stable Diffusion XL pipelines.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        print(f"Loading Stable Diffusion Pipeline: {self.model_id}...")        
        
        try:
            dtype = torch.float16 if self.device != "cpu" else torch.float32

            # Check if the model is XL or not by inspecting the config
            from diffusers import DiffusionPipeline
            config = DiffusionPipeline.load_config(self.model_id)
            pipeline_class_name = config.get("_class_name", "")
            is_xl = "XL" in pipeline_class_name
            PipelineClass = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

            # Load the full pipeline (UNet, VAE, Text Encoder, Scheduler)
            pipe = PipelineClass.from_pretrained(
                self.model_id, 
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
                safety_checker=None
            )
            
            if self.device == "cuda":
                pipe.enable_sequential_cpu_offload()
                if is_xl:
                    pipe.enable_vae_tiling()
            elif self.device == "mps":
                pipe.to("mps")
            else:
                pipe.to("cpu")
                
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