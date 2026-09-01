#!/usr/bin/env python3
"""Validate rendered GIFs and build contact sheets from actual GIF frames."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
FRAMES = (0, 50, 61, 120, 160)
METHODS = ("M1Y", "H1", "H2", "H5")
LABELS = {"M1Y": "M1-Y", "H1": "H1", "H2": "H2", "H5": "H5"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    destination = ROOT / "visual_audit"
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    font = ImageFont.load_default(size=14)
    for representation in "ABC":
        rows = []
        for method in METHODS:
            movie = (
                ROOT
                / f"movies/representation_{representation}"
                / f"movie_rep{representation}_{method}.gif"
            )
            sidecar = movie.with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if digest(movie) != metadata["movie_sha256"]:
                raise RuntimeError(f"movie hash mismatch: {movie}")
            with Image.open(movie) as image:
                frame_count = int(getattr(image, "n_frames", 1))
                duration_ms = int(image.info.get("duration", 0))
                if frame_count != 161 or duration_ms != 100:
                    raise RuntimeError(f"movie structure mismatch: {movie}")
                selected = []
                frame_statistics = {}
                for frame in FRAMES:
                    image.seek(frame)
                    rendered = image.convert("RGB")
                    array = np.asarray(rendered, dtype=np.float64)
                    frame_statistics[str(frame)] = {
                        "RGB_standard_deviation": float(np.std(array)),
                        "RGB_minimum": int(np.min(array)),
                        "RGB_maximum": int(np.max(array)),
                    }
                    rendered.thumbnail((360, 240), Image.Resampling.LANCZOS)
                    selected.append(rendered.copy())
                rows.append(selected)
            results.append(
                {
                    "representation": representation,
                    "method": method,
                    "movie": str(movie),
                    "movie_sha256": digest(movie),
                    "frame_count": frame_count,
                    "duration_ms": duration_ms,
                    "selected_frame_statistics": frame_statistics,
                    "all_selected_frames_nonblank": all(
                        row["RGB_standard_deviation"] > 5.0
                        for row in frame_statistics.values()
                    ),
                }
            )

        cell_width = max(frame.width for row in rows for frame in row)
        cell_height = max(frame.height for row in rows for frame in row)
        left = 70
        top = 32
        sheet = Image.new(
            "RGB",
            (left + len(FRAMES) * cell_width, top + len(METHODS) * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for column, frame in enumerate(FRAMES):
            label = f"{frame * 100} s"
            draw.text(
                (left + column * cell_width + 6, 8),
                label,
                fill="black",
                font=font,
            )
        for row_index, method in enumerate(METHODS):
            draw.text(
                (8, top + row_index * cell_height + 8),
                LABELS[method],
                fill="black",
                font=font,
            )
            for column, frame_image in enumerate(rows[row_index]):
                sheet.paste(
                    frame_image,
                    (left + column * cell_width, top + row_index * cell_height),
                )
        sheet_path = destination / f"movie_contact_sheet_rep{representation}.png"
        if sheet_path.exists():
            raise FileExistsError(f"refusing to overwrite {sheet_path}")
        sheet.save(sheet_path)

    record = {
        "status": "complete",
        "source": "frames decoded from final rendered GIFs",
        "selected_frames": list(FRAMES),
        "selected_times_s": [100 * frame for frame in FRAMES],
        "movie_count": len(results),
        "all_movies_have_161_frames": all(
            row["frame_count"] == 161 for row in results
        ),
        "all_movies_have_100ms_frames": all(
            row["duration_ms"] == 100 for row in results
        ),
        "all_selected_frames_nonblank": all(
            row["all_selected_frames_nonblank"] for row in results
        ),
        "movies": results,
    }
    record_path = destination / "MOVIE_VISUAL_AUDIT.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite {record_path}")
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("validated 12 GIFs and wrote three actual-frame contact sheets")


if __name__ == "__main__":
    main()
