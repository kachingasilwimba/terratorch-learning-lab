"""Encode a scene and compare spatially corresponding patch embeddings."""

from pathlib import Path

import numpy as np

from .data import BANDS, PATCH_SIZE, Scene, model_input, valid_pixels

MODEL = "prithvi_eo_v2_tiny_tl"
EMBEDDING_DIM = 192


def encode(
    scene: Scene,
    device: str = "cpu",
    pretrained: bool = True,
    checkpoint: Path | None = None,
    include_coordinates: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scene token and time-major grid of embedded patches."""
    import torch
    from terratorch.registry import BACKBONE_REGISTRY

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    if not pretrained:
        torch.manual_seed(12)

    count, _, height, width = scene.pixels.shape
    patch_rows, patch_columns = height // PATCH_SIZE, width // PATCH_SIZE
    model = BACKBONE_REGISTRY.build(
        MODEL,
        pretrained=pretrained,
        ckpt_path=str(checkpoint) if checkpoint else None,
        num_frames=count,
        bands=list(BANDS),
        out_indices=[-1],
    ).to(device).eval()
    inputs = torch.from_numpy(model_input(scene)).to(device)

    temporal_coords = location_coords = None
    if include_coordinates:
        years_days = [[day.year, day.timetuple().tm_yday] for day in scene.dates]
        temporal_coords = torch.tensor([years_days], dtype=torch.float32, device=device)
        if scene.location is not None:
            location_coords = torch.tensor([scene.location], dtype=torch.float32, device=device)

    with torch.inference_mode():
        tokens = model(inputs, temporal_coords=temporal_coords, location_coords=location_coords)[-1]
    expected = (1, 1 + count * patch_rows * patch_columns, EMBEDDING_DIM)
    if tuple(tokens.shape) != expected or not torch.isfinite(tokens).all().item():
        raise RuntimeError(f"Encoder output must be finite with shape {expected}. Got {tuple(tokens.shape)}.")

    tokens = tokens[0].cpu().numpy()
    patches = tokens[1:].reshape(count, patch_rows, patch_columns, EMBEDDING_DIM)
    return tokens[0], patches


def patch_distances(patches: np.ndarray, scene: Scene) -> np.ndarray:
    """Compare first and last dates with cosine distance, from 0 to 2."""
    if len(scene.dates) < 2:
        raise ValueError("At least two dates are required to compare embeddings.")
    first, last = patches[0], patches[-1]
    numerator = np.sum(first * last, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(last, axis=-1)
    count, patch_rows, patch_columns, _ = patches.shape
    valid = valid_pixels(scene.pixels).all(axis=1)
    valid = valid.reshape(count, patch_rows, PATCH_SIZE, patch_columns, PATCH_SIZE).any(axis=(2, 4))
    usable = (denominator > 0) & valid[0] & valid[-1]
    similarity = np.full_like(denominator, np.nan)
    np.divide(numerator, denominator, out=similarity, where=usable)
    return 1 - np.clip(similarity, -1, 1)
