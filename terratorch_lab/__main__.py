"""Run the TerraTorch learning lab from a terminal."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import BANDS, SCALE, Scene, download_sample, load_scene, model_input, ndvi, synthetic_scene, valid_pixels
from .encoder import MODEL, encode, patch_distances


def inspect(scene: Scene, output: Path) -> None:
    colors = np.moveaxis(scene.pixels[:, [2, 1, 0]], 1, -1)
    colors = np.where(valid_pixels(colors), colors, 0)
    colors = np.clip(colors / (0.30 * SCALE), 0, 1)
    vegetation = ndvi(scene.pixels)
    dates = len(scene.dates)
    figure, axes = plt.subplots(2, dates, figsize=(max(5, 3.2 * dates), 6.7), squeeze=False, layout="constrained")
    figure.suptitle(scene.source)
    for index, day in enumerate(scene.dates):
        axes[0, index].set_title(day.isoformat())
        axes[0, index].imshow(colors[index])
        layer = axes[1, index].imshow(vegetation[index], cmap="BrBG", vmin=-1, vmax=1)
    axes[0, 0].set_ylabel("Natural color")
    axes[1, 0].set_ylabel("NDVI")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(layer, ax=axes[1], shrink=0.75)
    path = output / "scene.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"{scene.source}\nBands {', '.join(BANDS)}\nModel input [B, C, T, H, W] {model_input(scene).shape}")
    print(f"Saved {path}")


def encode_scene(scene: Scene, output: Path, args: argparse.Namespace) -> None:
    cls_token, patch_tokens = encode(
        scene,
        device=args.device,
        pretrained=not args.random_weights,
        checkpoint=args.checkpoint,
        include_coordinates=not args.omit_coordinates,
    )
    weights = "random" if args.random_weights else "pretrained"
    path = output / "embeddings.npz"
    results = {
        "cls_token": cls_token,
        "patch_tokens": patch_tokens,
        "dates": [day.isoformat() for day in scene.dates],
        "model": MODEL,
        "weights": weights,
        "source": scene.source,
        "used_temporal_coordinates": not args.omit_coordinates,
        "used_location_coordinates": not args.omit_coordinates and scene.location is not None,
    }
    if len(scene.dates) > 1:
        results["cosine_distance"] = patch_distances(patch_tokens, scene)
    np.savez_compressed(path, **results)
    print(f"{MODEL} · {weights} weights · {args.device}")
    print(f"Scene token {cls_token.shape}\nPatch tokens [T, rows, columns, features] {patch_tokens.shape}")
    print(f"Saved {path}")

    if "cosine_distance" in results:
        figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
        distances = results["cosine_distance"]
        maximum = max(0.1, min(2.0, np.ceil(np.nanmax(distances, initial=0.0) * 10) / 10))
        layer = axis.imshow(distances, cmap="magma", vmin=0, vmax=maximum)
        axis.set_title(f"{scene.dates[0]} to {scene.dates[-1]} · {weights} weights")
        axis.set(xlabel="Patch column", ylabel="Patch row")
        figure.colorbar(layer, ax=axis, label="Cosine distance")
        path = output / "embedding-change.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("inspect", help="Plot RGB and NDVI for each date.")
    run = commands.add_parser("encode", help="Extract Prithvi-EO patch embeddings using TerraTorch.")
    sample = commands.add_parser("sample", help="Download a pair of public HLS sample rasters.")
    for command in (view, run, sample):
        command.add_argument("--output-dir", type=Path, default=Path("lab-output"))
    for command in (view, run):
        command.add_argument("--input", type=Path, help="A compatible .npz scene. Defaults to synthetic imagery.")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    weight_options = run.add_mutually_exclusive_group()
    weight_options.add_argument("--random-weights", action="store_true", help="Run an offline architecture exercise.")
    weight_options.add_argument("--checkpoint", type=Path, help="Path to a trusted Tiny-TL checkpoint file.")
    run.add_argument("--omit-coordinates", action="store_true", help="Skip temporal and location embeddings.")
    args = parser.parse_args()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "sample":
            path = args.output_dir / "hls-sample.npz"
            download_sample(path)
            print(f"Saved {path}")
            return

        scene = load_scene(args.input) if args.input else synthetic_scene()
        if args.command == "inspect":
            inspect(scene, args.output_dir)
        else:
            encode_scene(scene, args.output_dir, args)
    except ModuleNotFoundError as error:
        parser.exit(1, f"{error}. Install requirements-model.txt for encode or sample.\n")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
