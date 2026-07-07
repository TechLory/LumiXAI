"""Shared helpers for pixel-space attribution methods (image classification).

Unlike the text attributors, which all attribute to token embeddings, every image
classification attributor collapses a `[1, C, H, W]` attribution tensor down to a 2D
saliency map and packages it using the exact same `{"image_base64", "raw_matrix"}`
shape DAAM already produces for text-to-image tokens. This lets the frontend reuse its
existing image-heatmap overlay/canvas logic unchanged for the new modality.
"""

import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

GRID = 64


def collapse_pixel_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """Collapses a `[1, C, H, W]` pixel-attribution tensor into a `[H, W]` saliency map.

    Uses the sum of absolute values across channels rather than a signed sum. Adjacent
    color channels frequently have opposite-sign gradients at the same pixel (measured at
    ~53% of pixels on a real ResNet-18 example), so a signed sum cancels most of the real
    localized signal into near-zero salt-and-pepper noise. Summing magnitudes preserves
    "how much this pixel mattered" regardless of which channel carried the signal.
    """
    return attributions.squeeze(0).abs().sum(dim=0)


def render_image_heatmap(attributions: torch.Tensor, image: Image.Image) -> dict:
    """Builds the standardized heatmap payload for a pixel-space attribution map.

    Args:
        attributions (torch.Tensor): Raw attribution tensor of shape `[1, C, H, W]`,
            in the same spatial resolution the model's processor produced.
        image (PIL.Image.Image): The original (un-resized) input image, used as the
            background for the rendered overlay.

    Returns:
        dict: `{"image_base64": str, "raw_matrix": List[List[float]]}`, matching the
            per-token heatmap entries DAAM emits for text-to-image models.
    """
    heatmap_2d = collapse_pixel_attributions(attributions)

    resized = F.interpolate(
        heatmap_2d.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(GRID, GRID),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    raw_matrix = resized.detach().cpu().tolist()

    heatmap_np = heatmap_2d.detach().to(torch.float32).cpu().numpy()
    vmin, vmax = np.percentile(heatmap_np, 1), np.percentile(heatmap_np, 99)
    normalized = np.clip((heatmap_np - vmin) / (vmax - vmin + 1e-8), 0, 1)

    w, h = image.size
    fig = Figure(figsize=(6.0, 6.0 * h / w))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, extent=(0, w, h, 0))
    ax.imshow(normalized, cmap="jet", alpha=0.6, extent=(0, w, h, 0))
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "raw_matrix": raw_matrix,
    }


def image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image as a base64 PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_base64_image(data: str) -> Image.Image:
    """Decodes a base64 (optionally data-URI prefixed) string into an RGB PIL Image."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(BytesIO(raw)).convert("RGB")


def build_patch_feature_mask(height: int, width: int, patch_size: int, device: str) -> torch.Tensor:
    """Builds a `[1, 1, H, W]` grid of integer patch ids for Occlusion/LIME feature masks.

    Broadcasting this over the channel dimension makes every channel within a spatial
    patch share one "feature", so perturbations move whole patches instead of individual
    pixels or channels.
    """
    n_rows = (height + patch_size - 1) // patch_size
    n_cols = (width + patch_size - 1) // patch_size
    patch_ids = torch.arange(n_rows * n_cols, device=device).reshape(n_rows, n_cols)
    full = patch_ids.repeat_interleave(patch_size, dim=0)[:height, :].repeat_interleave(patch_size, dim=1)[:, :width]
    return full.unsqueeze(0).unsqueeze(0)
