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
        image (PIL.Image.Image): The *preprocessed* image the model saw (see
            `denormalize_pixel_values`), so the overlay aligns with the attribution grid.

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


def denormalize_pixel_values(pixel_values: torch.Tensor, image_mean, image_std) -> Image.Image:
    """Reconstructs a displayable RGB PIL image from the model's normalized pixel tensor.

    Every image attributor operates in the processor's *normalized pixel space* — the
    `[1, C, H, W]` tensor after resize/crop/rescale/normalize — and the attribution grid
    lives in that same space. Overlaying the heatmap on the user's original upload is
    therefore only correct when the processor did a plain aspect-preserving resize; any
    center-crop, squish, or letterbox shifts the pixels relative to the map.

    Reversing the normalize (and rescale) step here yields exactly the image the model
    saw, so the overlay aligns pixel-for-pixel with the attribution for *any* processor,
    with no per-model configuration needed. `image_mean`/`image_std` may be None (e.g. a
    processor with `do_normalize=False`), in which case the tensor is already in the
    rescaled [0, 1] range and only clamped.

    Args:
        pixel_values (torch.Tensor): The processor's `[1, C, H, W]` normalized tensor.
        image_mean: Per-channel mean used for normalization, or None.
        image_std: Per-channel std used for normalization, or None.

    Returns:
        PIL.Image.Image: The de-normalized image the model actually received.
    """
    tensor = pixel_values.detach().to(torch.float32).cpu().squeeze(0)  # [C, H, W]
    if image_mean is not None and image_std is not None:
        mean = torch.tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(-1, 1, 1)
        tensor = tensor * std + mean

    array = tensor.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:  # single-channel models: drop the trailing dim for PIL
        array = array[:, :, 0]
    return Image.fromarray(array).convert("RGB")


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


def build_superpixel_feature_mask(image: Image.Image, n_segments: int, device: str) -> torch.Tensor:
    """Builds a `[1, 1, H, W]` grid of SLIC superpixel ids for LIME's feature mask.

    Captum's LIME vision tutorial groups pixels into *segmentation* superpixels rather
    than a fixed grid, so each perturbed "feature" is a coherent, edge-following region
    (an object part) instead of an arbitrary square. Perturbing whole superpixels makes
    the resulting attribution align with real boundaries in the image, which is both what
    the tutorial does and what produces clean, interpretable maps.

    Runs SLIC on the *preprocessed* (de-normalized) image so the returned ids line up
    spatially with the model's `[1, C, H, W]` pixel tensor. The `[1, 1, H, W]` shape
    broadcasts over the channel dimension, so every channel of a superpixel is perturbed
    together, exactly like `build_patch_feature_mask`.

    Args:
        image (PIL.Image.Image): The preprocessed image the model saw (see
            `denormalize_pixel_values`).
        n_segments (int): Approximate number of superpixels SLIC should produce; the
            actual count varies with image content.
        device (str): Device to place the returned mask tensor on.

    Returns:
        torch.Tensor: A `[1, 1, H, W]` long tensor of contiguous superpixel ids.
    """
    from skimage.segmentation import slic

    array = np.asarray(image).astype(np.float32) / 255.0  # [H, W, C] in [0, 1]
    segments = slic(array, n_segments=n_segments, compactness=10.0, start_label=0)
    mask = torch.as_tensor(segments, dtype=torch.long, device=device)
    return mask.unsqueeze(0).unsqueeze(0)


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
