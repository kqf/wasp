import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np
from dataclasses_json import dataclass_json


def iterate(
    path: Path,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        print(f"Error: Could not open video {path}")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            break  # Properly exit loop when the video ends

        height, width, _ = frame.shape
        mid = width // 2
        limage = frame[:, :mid, :]
        rimage = frame[:, mid:, :]

        yield limage, rimage

    capture.release()


@dataclass_json
@dataclass
class VideoSample:
    name: str
    distance: float
    bbox: tuple[float, float, float, float]


def load_sample(filename: str) -> VideoSample:
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    return VideoSample.from_dict(data)  # type: ignore


def draw_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float]):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center = (x + w // 2, y + h // 2)
    cv2.circle(image, center, 3, (0, 0, 255), -1)


def main():
    sample = load_sample("datasets/distances/samples.json")
    tracker = cv2.TrackerCSRT_create()
    bbox = sample.bbox

    for limage, rimage in iterate(sample.name):
        if bbox is not None:
            tracker.init(limage, bbox)
            bbox = None

        success, new_bbox = tracker.update(limage)

        draw_bbox(limage, new_bbox)
        cv2.imshow("Left Frame", limage)
        cv2.imshow("Right Frame", rimage)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
