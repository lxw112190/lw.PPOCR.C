from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    image = Image.open(arguments.input).convert("RGB")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{image.width} {image.height}\n255\n".encode("ascii")
    arguments.output.write_bytes(header + image.tobytes())


if __name__ == "__main__":
    main()
