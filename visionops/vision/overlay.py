"""Readable track overlays shared by the retail runner and generic video jobs."""
import cv2

# Distinct, saturated BGR tones that stay legible on bright retail footage.
TRACK_PALETTE = [(255, 178, 29), (49, 210, 207), (90, 104, 255), (199, 116, 255), (72, 200, 120),
                 (31, 148, 255), (180, 105, 56), (147, 69, 245), (0, 204, 255), (170, 170, 255)]


def track_color(track_id: int | None) -> tuple[int, int, int]:
    return (0, 230, 255) if track_id is None else TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def label_chip(image, label: str, anchor: tuple[int, int], color, scale: float = 0.52) -> None:
    """Draw a filled label chip above `anchor`, kept inside the frame."""
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 4
    x = max(0, min(anchor[0], image.shape[1]-width-2*pad))
    y = anchor[1]-height-baseline-2*pad
    if y < 0:
        y = anchor[1]
    y = min(y, image.shape[0]-height-baseline-2*pad)
    cv2.rectangle(image, (x, y), (x+width+2*pad, y+height+baseline+2*pad), color, -1)
    text = (20, 20, 20) if sum(color) > 420 else (255, 255, 255)
    cv2.putText(image, label, (x+pad, y+pad+height), cv2.FONT_HERSHEY_SIMPLEX, scale, text, 1, cv2.LINE_AA)


def draw_track(image, bbox, track_id: int | None, confidence: float | None = None) -> tuple[int, int, int]:
    x1, y1, x2, y2 = map(int, bbox)
    color = track_color(track_id)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"ID {track_id if track_id is not None else '?'}"
    if confidence is not None:
        label += f"  {confidence:.0%}"
    label_chip(image, label, (x1, y1), color)
    return color
