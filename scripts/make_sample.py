"""Generate a synthetic dark night scene that imitates a camera trap photo.

The real ``deer.jpg`` from the project slide is not part of the repository,
so the demo and tests use this reproducible image instead. Every pixel lies
roughly in the 0-40 range, which gives the same left-packed histogram as a
real underexposed night photo.

Usage:
    python scripts/make_sample.py [--output PATH] [--seed N]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sample" / "night_deer.png"

HEIGHT, WIDTH = 480, 640
HORIZON_Y = 330
MAX_VALUE = 40


def _draw_background(rng: np.random.Generator) -> np.ndarray:
    """Dark forest background with a slightly brighter ground below the horizon."""
    img = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    # Forest depth behind the trees: a soft vertical gradient (values ~10-14).
    img[:HORIZON_Y, :] = np.linspace(10, 14, HORIZON_Y, dtype=np.float32)[:, None]
    # Ground lit by the infrared flash: brighter towards the camera (~16-24).
    img[HORIZON_Y:, :] = np.linspace(16, 24, HEIGHT - HORIZON_Y, dtype=np.float32)[:, None]

    # Grass blades: short, slightly tilted strokes on the ground.
    for _ in range(900):
        x = int(rng.integers(0, WIDTH))
        y = int(rng.integers(HORIZON_Y + 5, HEIGHT))
        length = int(rng.integers(6, 18))
        tilt = int(rng.integers(-4, 5))
        value = float(rng.uniform(22, 30))
        cv2.line(img, (x, y), (x + tilt, y - length), value, 1, cv2.LINE_AA)
    return img


def _draw_trees(img: np.ndarray, rng: np.random.Generator) -> None:
    """Dark vertical tree trunks that reach from the top of the frame to the ground."""
    for _ in range(9):
        x = int(rng.integers(0, WIDTH))
        width = int(rng.integers(10, 30))
        bottom = HORIZON_Y + int(rng.integers(0, 40))
        value = float(rng.uniform(3, 7))
        cv2.rectangle(img, (x, 0), (x + width, bottom), value, thickness=-1)
        # A few thin side branches.
        for _ in range(int(rng.integers(1, 4))):
            by = int(rng.integers(20, HORIZON_Y - 60))
            direction = 1 if rng.random() < 0.5 else -1
            end = (x + width // 2 + direction * int(rng.integers(30, 70)), by - int(rng.integers(20, 50)))
            cv2.line(img, (x + width // 2, by), end, value, 3, cv2.LINE_AA)


def _draw_deer(img: np.ndarray) -> None:
    """A simple deer silhouette, slightly brighter than its surroundings."""
    value = 32.0
    body_center = (330, 300)
    cv2.ellipse(img, body_center, (85, 38), 0, 0, 360, value, thickness=-1)
    # Legs.
    for leg_x in (265, 285, 370, 390):
        cv2.line(img, (leg_x, 320), (leg_x, 400), value, 9, cv2.LINE_AA)
    # Neck and head.
    neck = np.array([[380, 290], [405, 285], [430, 205], [412, 200]], dtype=np.int32)
    cv2.fillPoly(img, [neck], value)
    cv2.ellipse(img, (432, 198), (26, 14), -20, 0, 360, value, thickness=-1)
    # Ears.
    cv2.ellipse(img, (412, 182), (12, 5), -50, 0, 360, value, thickness=-1)
    # Antlers.
    antler_segments = [
        ((420, 188), (405, 140)), ((405, 140), (390, 110)), ((405, 140), (425, 118)),
        ((395, 120), (378, 112)), ((428, 186), (450, 140)), ((450, 140), (462, 108)),
        ((450, 140), (472, 130)), ((458, 120), (445, 104)),
    ]
    for start, end in antler_segments:
        cv2.line(img, start, end, value, 4, cv2.LINE_AA)
    # Tail.
    cv2.ellipse(img, (246, 288), (10, 6), 30, 0, 360, value, thickness=-1)
    # Eye reflecting the infrared flash: the brightest spot in the scene.
    cv2.circle(img, (442, 194), 3, float(MAX_VALUE), thickness=-1)


def make_night_scene(seed: int = 376) -> np.ndarray:
    """Return a dark uint8 grayscale night scene of shape (480, 640)."""
    rng = np.random.default_rng(seed)
    img = _draw_background(rng)
    _draw_trees(img, rng)
    _draw_deer(img)
    # Sensor noise, which is typical for high-ISO night shots.
    img += rng.normal(0.0, 2.0, size=img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, MAX_VALUE).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic dark camera trap image.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output image path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})")
    parser.add_argument("--seed", type=int, default=376, help="random seed (default: 376)")
    args = parser.parse_args()

    img = make_night_scene(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), img):
        raise SystemExit(f"Failed to write image to {args.output}")
    print(f"Saved {args.output} ({img.shape[1]}x{img.shape[0]}, "
          f"pixel range {img.min()}-{img.max()}, mean {img.mean():.1f})")


if __name__ == "__main__":
    main()
