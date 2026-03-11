import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict
from diffusers.models.attention_processor import Attention

class CustomDAAMHeatmap:
    """Manage per-token heatmaps"""
    def __init__(self, heatmap: torch.Tensor):
        self.heatmap = heatmap

    def plot_overlay(self, image):
        w, h = image.size
        heatmap_np = self.heatmap.cpu().numpy()
        
        # Test: percentile normalization 1-99 instead of min-max (cut off outliers)
        vmin = np.percentile(heatmap_np, 1)
        vmax = np.percentile(heatmap_np, 99)
        
        heatmap_np = np.clip((heatmap_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
        #heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.imshow(heatmap_np, cmap='jet', alpha=0.6, extent=(0, w, h, 0))
        ax.axis('off')
        return fig

class CaptureAttnProcessor:
    def __init__(self):
        self.attention_probs = []

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # 1. Setup Input
        batch_size, sequence_length, _ = hidden_states.shape
        # Cross-Attention or Self-Attention
        is_cross = encoder_hidden_states is not None
        context = encoder_hidden_states if is_cross else hidden_states
        
        heads = attn.heads
        dim_head = attn.to_q.out_features // heads

        # 2. Projections & Manual Reshape (Batch*Heads)
        query = attn.to_q(hidden_states)
        key = attn.to_k(context)
        value = attn.to_v(context)

        query = query.view(batch_size, -1, heads, dim_head).transpose(1, 2).reshape(batch_size * heads, -1, dim_head)
        key = key.view(batch_size, -1, heads, dim_head).transpose(1, 2).reshape(batch_size * heads, -1, dim_head)
        value = value.view(batch_size, -1, heads, dim_head).transpose(1, 2).reshape(batch_size * heads, -1, dim_head)

        # 3. Attention Scores
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * scale
        attn_probs = attn_scores.softmax(dim=-1)

        # 4. Capture Hook
        if is_cross:
            self.attention_probs.append(attn_probs.detach().cpu())

        # 5. Output
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = hidden_states.reshape(batch_size, heads, -1, dim_head).transpose(1, 2).reshape(batch_size, -1, heads * dim_head)
        hidden_states = attn.to_out[0](hidden_states) # Linear projection
        hidden_states = attn.to_out[1](hidden_states) # Linear projection
        return hidden_states

class trace:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.original_processors = {}
        self.capture_processor = CaptureAttnProcessor()

    def __enter__(self):
        unet = self.pipeline.unet
        for name, module in unet.named_modules():
            if name.endswith("attn2") and hasattr(module, "set_processor"):
                self.original_processors[name] = module.processor
                module.set_processor(self.capture_processor)
        return self

    def __exit__(self, exc_type, exc_value, traceback): #
        unet = self.pipeline.unet
        for name, module in unet.named_modules():
            if name in self.original_processors:
                module.set_processor(self.original_processors[name])

    def compute_heat_maps(self) -> Dict[int, CustomDAAMHeatmap]:
        """
        Return a dictionary of {token_index: CustomDAAMHeatmap}
        instead of a single global heatmap.
        """
        if not self.capture_processor.attention_probs:
            raise ValueError("No attention captured.")

        all_maps = self.capture_processor.attention_probs

        valid_maps = []
        for m in all_maps:
            # m shape: [Batch*Heads, Pixels, Tokens]
            num_pixels = m.shape[1]
            
            # 16*16 ---- 64*64
            if 256 <= num_pixels <= 4096:
                valid_maps.append(m)

        if not valid_maps:
            print("No valid attention maps found. Using all captured maps.")
            valid_maps = all_maps

        num_tokens = valid_maps[0].shape[-1]
        
        target_dim = 64 
        accumulated_maps = torch.zeros(target_dim, target_dim, num_tokens)
        count = 0

        for m in valid_maps:
            # Shape: [Pixels, Tokens]
            m_avg_heads = m.mean(dim=0) 
            
            # 2. Reshape in 2D
            side = int(np.sqrt(m_avg_heads.shape[0]))
            if side * side != m_avg_heads.shape[0]: continue
            
            m_2d = m_avg_heads.view(side, side, num_tokens)
            
            # 3. Upscale Bicubico a (64x64)
            m_permuted = m_2d.permute(2, 0, 1).unsqueeze(1) 
            
            m_resized = torch.nn.functional.interpolate(
                m_permuted.to(torch.float32), 
                size=(target_dim, target_dim), 
                mode='bicubic', 
                align_corners=False
            ).to(m_permuted.dtype)
            
            accumulated_maps += m_resized.squeeze(1).permute(1, 2, 0)
            count += 1

        if count > 0:
            accumulated_maps /= count

        # build Output
        results = {}
        for token_idx in range(num_tokens):
            token_map = accumulated_maps[:, :, token_idx]
            if token_map.max() > 0:
                results[token_idx] = CustomDAAMHeatmap(token_map)
                
        return results