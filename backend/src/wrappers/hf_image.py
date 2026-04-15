import torch
from typing import Any, Union, List
from ..abstract import BaseWrapper
from ..utils.hf_auth import hf_auth_kwargs

class HFImageWrapper(BaseWrapper):
    """Wrapper for Text-to-Image Generation models using Hugging Face Diffusers.
    
    This class supports both standard Stable Diffusion architectures and Stable Diffusion XL.
    It automatically detects the model type from its configuration and loads the appropriate 
    pipeline, handling precision and memory optimizations based on the selected hardware.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        """Initializes the wrapper and triggers model loading.

        Args:
            model_id (str): The Hugging Face Hub ID (e.g., "stabilityai/sd-turbo") or local path.
            device (str, optional): The target device ("cpu", "cuda", "mps"). Defaults to "cpu".
        """
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        """Loads the Stable Diffusion pipeline and its components (UNet, VAE, Text Encoder).

        Automatically detects if the model is an XL variant to use the `StableDiffusionXLPipeline` 
        and applies VAE tiling for memory optimization on CUDA devices.

        Returns:
            Any: The loaded Hugging Face Diffusers pipeline.

        Raises:
            RuntimeError: If the pipeline fails to load or download from the Hub.
        """
        print(f"Loading Stable Diffusion Pipeline: {self.model_id}...")        
        
        try:
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            auth_kwargs = hf_auth_kwargs()

            from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
            config = DiffusionPipeline.load_config(self.model_id, **auth_kwargs)
            pipeline_class_name = config.get("_class_name", "")
            is_xl = "XL" in pipeline_class_name
            PipelineClass = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

            pipe = PipelineClass.from_pretrained(
                self.model_id, 
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
                safety_checker=None,
                **auth_kwargs,
            )
            
            if self.device.startswith("cuda"):
                # !! -- TO AVOID ERROR: RuntimeError: Tensor on device meta is not on the expected device cuda:0! -- !!
                # Moving the entire pipeline to GPU. Ensure to have enough VRAM for the model you are loading!
                pipe.to(self.device)
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
        """Executes the standard generation pipeline to produce an image.

        Args:
            input_data (str): The text prompt describing the desired image.

        Returns:
            Any: A PIL Image object representing the first generated image.

        Raises:
            ValueError: If the provided `input_data` is not a string.
        """
        if not isinstance(input_data, str):
            raise ValueError("Input for HFImageWrapper must be a text prompt string.")

        print(f"Generating image for prompt: '{input_data}'...")
        
        with torch.no_grad():
            output = self.model(input_data, num_inference_steps=30)
            
        return output.images[0]

    def get_embedding_layer(self) -> torch.nn.Module:
        """Retrieves the input embedding layer of the Text Encoder (CLIP).

        Required for token-level gradient attribution analysis.

        Returns:
            torch.nn.Module: The PyTorch embedding layer.
        """
        return self.model.text_encoder.get_input_embeddings()
