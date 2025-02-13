import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np
from dataclasses_json import dataclass_json
from uncertainties import ufloat


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

    @property
    def meters(self) -> float:
        # Convert distance in inches to meters
        return self.distance * 0.0254


def load_sample(filename: str) -> list[VideoSample]:
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    return [VideoSample.from_dict(sample) for sample in data]  # type: ignore


def bcenter(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def draw_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float]):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center = tuple(map(int, bcenter(bbox)))
    cv2.circle(image, center, 3, (0, 255, 0), -1)
    cv2.circle(image, center, max(w, h) // 8, (0, 255, 0), 2)
    return image


def draw_overlay(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    scale_factor: int = 3,
):
    x, y, w, h = map(int, bbox)
    patch = image[y : y + h, x : x + w]
    if patch.size > 0:
        enlarged_patch = cv2.resize(
            patch,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_LINEAR,
        )

        h_img, w_img, _ = image.shape
        patch_h, patch_w, _ = enlarged_patch.shape
        y_offset = h_img - patch_h - 10
        x_offset = w_img - patch_w - 10

        image[
            y_offset : y_offset + patch_h, x_offset : x_offset + patch_w
        ] = enlarged_patch

    return image


def find_match(
    limage: np.ndarray,
    rimage: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    x, y, w, h = map(int, bbox)
    template = limage[y : y + h, x : x + w]

    if template.size == 0:
        raise RuntimeError("Got the empty template size")

    res = cv2.matchTemplate(rimage, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    match_x, match_y = max_loc
    return match_x, match_y, w, h


def compute_distance(
    lpoint: tuple[float, float],
    rpoint: tuple[float, float],
    focal_length,
    focal_error=0,
    baseline=0,
    baseline_error=0,
    disparity_error=0.0,
):
    disparity = ufloat(abs(lpoint[0] - rpoint[0]), disparity_error)
    if disparity.nominal_value <= 0:
        raise ValueError("Disparity is zero or negative!")

    f = ufloat(focal_length, focal_error)
    B = ufloat(baseline, baseline_error)
    return (f * B) / disparity


def compute_distance_acc(
    lpoint: tuple[float, float],
    rpoint: tuple[float, float],
    focal_length,
    focal_error=0,
    baseline=1.0,
    baseline_error=0,
    disparity_error=0.0,
):
    disparity_x = ufloat(lpoint[0] - rpoint[0], disparity_error)
    disparity_y = ufloat(lpoint[1] - rpoint[1], disparity_error)
    disparity = (disparity_x**2 + disparity_y**2) ** 0.5

    if disparity_x.nominal_value <= 0:
        raise ValueError(
            "Horizontal disparity is zero or negative! Check camera setup."
        )

    f = ufloat(focal_length, focal_error)
    B = ufloat(baseline, baseline_error)
    return (f * B) / disparity


def main():
    sample = load_sample("datasets/distances/samples.json")[0]
    tracker = cv2.TrackerMIL.create()
    loriginal = sample.bbox

    for limage, rimage in iterate(sample.name):
        if loriginal is not None:
            tracker.init(limage, list(map(int, loriginal)))
            loriginal = None

        success, lbbox = tracker.update(limage)
        if not success:
            continue

        limage = draw_bbox(limage, lbbox)
        limage = draw_overlay(limage, lbbox)
        rbbox = find_match(limage, rimage, lbbox)
        rimage = draw_bbox(rimage, rbbox)
        rimage = draw_overlay(rimage, rbbox)

        lcetner = bcenter(lbbox)
        rcenter = bcenter(rbbox)

        print("Frame shape:", limage.shape)
        print("Points", lcetner, rcenter)
        print(
            "Actual distance",
            sample.distance,
            "in inches",
            sample.meters,
            "meters",
        )

        # baseline is 45 mm
        dist = compute_distance(
            lcetner,
            rcenter,
            focal_length=1765.3463,
            focal_error=1,
            baseline=0.025,
            baseline_error=0.0001,
        )
        dist_vertical_disparity = compute_distance_acc(
            lcetner,
            rcenter,
            focal_length=1765.3463,
            focal_error=1,
            baseline=0.025,
            baseline_error=0.0001,
        )

        print(
            sample.meters,
            dist,
            dist_vertical_disparity,
            dist / sample.meters,
        )

        cv2.imshow("Left Frame", np.hstack((limage, rimage)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
