"""Extract auditable sequences without rerunning detection or altering track IDs."""
from pathlib import Path
import json

import cv2
import numpy as np


def main() -> None:
    output = Path("outputs/retail_store_test")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    track_rows = [json.loads(line) for line in (output / "tracks.jsonl").read_text().splitlines()]
    events = json.loads((output / "events.json").read_text())["events"]
    indices = {0, 40, 47, 49, 55, 131, 263, 395, 659, 923, 1187, 1451}
    for event in events:
        if "line_name" in event:
            indices.update(max(0, min(summary["source"]["decoded_frames"]-1, event["frame_index"]+offset)) for offset in (-8, 0, 8))
    source = cv2.VideoCapture(summary["source"]["path"])
    annotated = cv2.VideoCapture(str(output / "annotated.mp4"))
    directory = output / "review_frames"
    directory.mkdir(exist_ok=True)
    sheets = {}
    for kind, cap in [("source", source), ("annotated", annotated)]:
        tiles = []
        for index in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode review frame {index}")
            cv2.imwrite(str(directory / f"frame_{index:06d}_{kind}.jpg"), frame)
            tile = cv2.resize(frame, (635, 360))
            cv2.rectangle(tile, (0, 0), (635, 25), (0, 0, 0), -1)
            cv2.putText(tile, f"{index} / {track_rows[index]['timestamp']:.2f}s", (8, 18), 0, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
        cap.release()
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
        for part in range(0, len(rows), 4):
            filename = f"review_{kind}_{part//4+1}.jpg"
            cv2.imwrite(str(output / filename), np.vstack(rows[part:part+4]))
            sheets[filename] = sorted(indices)[part*3:(part+4)*3]
    (output / "review_contact_sheets.json").write_text(json.dumps(sheets, indent=2), encoding="utf-8")
    print(json.dumps({"review_frames": len(indices), "contact_sheets": sheets}))


if __name__ == "__main__":
    main()
