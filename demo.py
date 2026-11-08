"""Week 1 demo: global histogram equalization of a dark camera trap image.

Produces a 2x2 figure: before/after images on top, their 0-255 histograms
below. Runs headless (matplotlib Agg backend), no display needed.

Usage:
    python scripts/make_sample.py      # creates the default synthetic input
    python demo.py                     # or: python demo.py --input deer.jpg
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from src.enhancement import equalize_histogram, load_grayscale  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "data" / "sample" / "night_deer.png"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "week1_comparison.png"

BEFORE_COLOR = "#6aa56a"
AFTER_COLOR = "#e0a040"


def describe(img: np.ndarray) -> str:
    return f"min={img.min():3d}  max={img.max():3d}  mean={img.mean():6.1f}  std={img.std():5.1f}"


def plot_histogram(ax: plt.Axes, img: np.ndarray, color: str, title: str) -> None:
    counts = np.bincount(img.ravel(), minlength=256)
    ax.bar(np.arange(256), counts, width=1.0, color=color)
    ax.set_xlim(0, 255)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_yticks([])
    ax.set_xlabel("Pixel intensity")
    ax.set_title(title, color=color, fontweight="bold")


def save_comparison(before: np.ndarray, after: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5),
                             gridspec_kw={"height_ratios": [3, 1.4]})
    for ax, img, title, color in (
        (axes[0, 0], before, "Before (original)", BEFORE_COLOR),
        (axes[0, 1], after, "After (equalized)", AFTER_COLOR),
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color=color, fontweight="bold")
        ax.axis("off")

    plot_histogram(axes[1, 0], before, BEFORE_COLOR, "Histogram before")
    plot_histogram(axes[1, 1], after, AFTER_COLOR, "Histogram after")

    fig.suptitle("Camera Trap Image Enhancement - Histogram Equalization", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Histogram equalization demo for dark camera trap images.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"input image (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"comparison figure (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})")
    args = parser.parse_args(argv)

    try:
        before = load_grayscale(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.input == DEFAULT_INPUT:
            print("Hint: run 'python scripts/make_sample.py' first, or pass --input <image>.", file=sys.stderr)
        return 1

    after = equalize_histogram(before)
    save_comparison(before, after, args.output)

    print(f"Input : {args.input}")
    print(f"Before: {describe(before)}")
    print(f"After : {describe(after)}")
    print(f"Saved comparison figure to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
