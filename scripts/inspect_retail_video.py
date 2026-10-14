"""Prepare the requested input location and a source contact sheet for geometry review."""
from pathlib import Path
import json
import shutil

import cv2
import numpy as np


def main() -> None:
    source = Path("data/videos/retail_store.mp4")
    if not source.exists():
        original = Path("data/retail_store.mp4")
        if not original.exists():
            raise FileNotFoundError(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, source)
    output = Path("outputs/retail_store_test")
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {source}")
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    metadata = {"source": source.as_posix(), "frames": frames, "fps": fps,
                "width": int(cap.get(3)), "height": int(cap.get(4)), "duration_seconds": frames / fps}
    tiles = []
    for index in np.linspace(0, frames - 1, 9).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {index}")
        if index == 0:
            cv2.imwrite(str(output / "source_frame_0000.jpg"), frame)
        tile = cv2.resize(frame, (635, 360))
        cv2.rectangle(tile, (0, 0), (635, 26), (0, 0, 0), -1)
        cv2.putText(tile, f"Frame {index} | {index / fps:.2f}s", (10, 19), 0, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
    cap.release()
    cv2.imwrite(str(output / "source_contact_sheet.jpg"), np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, 9, 3)]))
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
