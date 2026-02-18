import torch
import torch.nn.functional as F
from typing import Any, Dict, Union, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..abstract import BaseWrapper

class HFTextGenerationWrapper(BaseWrapper):
    """
    Wrapper for Causal Language Models (e.g., GPT-2, Llama, Gemma).
    Uses AutoModelForCausalLM from HuggingFace.
    Supports multi-token generation and probability extraction.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__(model_id, device)

    def load_model(self) -> Any:
        print(f"Loading HF Generation Model: {self.model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        model.to(self.device)
        model.eval()
        return model

    def generate_text(self, prompt: str, max_new_tokens: int = 20) -> Tuple[str, List[str], List[float]]:
        """
        Generates a sequence of text starting from the prompt.
        Returns:
        1. Full generated text string
        2. List of generated token strings
        3. List of probabilities (0.0 - 1.0) for each generated token
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Generate token IDs
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,  # Deterministic for reproducibility in XAI
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract only the new tokens
        generated_ids = output_ids[0][input_length:]
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Re-run forward pass to get probabilities for the generated sequence
        probs = []
        tokens_text = []

        # Step-by-step iteration to extract probability of the chosen token
        current_input_ids = inputs.input_ids        
        for token_id in generated_ids:
            with torch.no_grad():
                outputs = self.model(current_input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Get prob of the actual token chosen
                token_prob = next_token_probs[0, token_id].item()
                probs.append(token_prob)
                
                # Decode single token
                token_str = self.tokenizer.decode([token_id])
                tokens_text.append(token_str)
                
                # Append to input for next step
                current_input_ids = torch.cat([current_input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)

        return full_text, tokens_text, probs

    def generate(self, input_data: Union[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Input: Text string or tokenized inputs
        Output: Logits for next token [Batch, VocabSize]
        """
        # If a string is provided, tokenize it
        if isinstance(input_data, str):
            inputs = self.tokenizer(
                input_data, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
        else:
            inputs = input_data

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract logits only for the last token in the sequence
        # outputs.logits shape is [Batch, SeqLen, VocabSize]
        next_token_logits = outputs.logits[:, -1, :]
        
        return next_token_logits
    
    def forward_func(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor: # type: ignore
        """
        Forward function compatible with Captum.
        Takes embeddings, returns logits for the LAST token position.
        """
        # Captum passes inputs_embeds directly
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        # Return logits of the last token (next-token prediction)
        return outputs.logits[:, -1, :]

    def get_embedding_layer(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()