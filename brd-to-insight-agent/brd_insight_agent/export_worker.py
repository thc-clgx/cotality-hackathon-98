"""One-shot renderer launched by filename, never through brd_insight_agent.__init__."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def render(request_path: Path) -> None:
    import plotly.io as pio

    request = json.loads(request_path.read_text(encoding="utf-8"))
    figures = request["figures"]
    if not isinstance(figures, list) or not figures:
        raise ValueError("The render request contains no figures.")
    if not callable(getattr(pio, "write_images", None)):
        raise RuntimeError("Batch export requires Plotly >= 6.1 and Kaleido >= 1 with Chrome installed.")
    image_format = request.get("format", "png")
    if image_format not in {"png", "svg", "jpeg"}:
        raise ValueError("Unsupported image format.")
    paths = [request_path.parent / f"{index:02d}.{image_format}" for index in range(1, len(figures) + 1)]
    pio.defaults.mathjax = None
    pio.write_images(
        figures,
        paths,
        format=image_format,
        width=request["width"],
        height=request["height"],
        scale=request["scale"],
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("Expected the absolute path to a render request.", file=sys.stderr)
        return 2
    try:
        request_path = Path(arguments[0])
        if not request_path.is_absolute():
            raise ValueError("The render request path must be absolute.")
        render(request_path)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())