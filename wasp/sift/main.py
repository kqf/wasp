from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def dump_database(output: str, descriptors) -> None:
    np.savez_compressed(output, descriptors=descriptors)


def build_features(impath: str) -> list[np.ndarray]:
    path = Path(impath)
    sift = cv2.SIFT_create(contrastThreshold=0.04)

    db = []
    for file in path.glob("*.jpg"):
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        mask = np.load(file.with_suffix(".npy"))
        if "4.jpg" in str(file):
            continue
        _, descriptors = sift.detectAndCompute(image, mask)
        if descriptors is not None:
            db.append(descriptors)

    return np.vstack(db)


@dataclass
class Detection:
    class_name: str
    num_matches: int
    matched_locations: list[tuple]


def detect_objects(
    input_image: np.ndarray,
    stacked_databases: dict[str, np.ndarray],
    match_threshold: int = 30,
) -> list[Detection]:

    sift = cv2.SIFT_create(contrastThreshold=0.04)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    keypoints, descriptors = sift.detectAndCompute(input_image, None)
    if descriptors is None:
        return []

    detections: list[Detection] = []

    for class_name, class_descriptors in stacked_databases.items():
        matches = matcher.match(descriptors, class_descriptors)
        if len(matches) > match_threshold:
            matched_locations = [keypoints[m.queryIdx].pt for m in matches]
            detections.append(
                Detection(
                    class_name,
                    len(matches),
                    matched_locations,
                    # matches,
                )
            )

    return detections


def visualize_detections(
    input_image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    for detection in detections:
        for x, y in detection.matched_locations:
            cv2.circle(output_image, (int(x), int(y)), 5, (0, 255, 0), 2)
        cv2.putText(
            output_image,
            f"{detection.class_name}: {detection.num_matches} matches",
            (10, 30 + 30 * detections.index(detection)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output_image


def main():
    stacked_databases = {
        "A": build_features("./datasets/a/"),
        # "B": build_features("./B"),
        # "C": build_features("./C"),
    }

    for name, base in stacked_databases.items():
        dump_database(f"database-{name}.npy", base)

    # Prepare the output folder
    outpath = Path("datasets/test/v1-SIFT-masks")
    outpath.mkdir(parents=True, exist_ok=True)

    for file in Path("datasets/test").glob("*.jpg"):
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        detections = detect_objects(image, stacked_databases)
        # Visualize and save
        annotated = visualize_detections(image, detections)

        cv2.imwrite(str(outpath / file.name), annotated)
        cv2.imshow("output", annotated)
        cv2.waitKey()


if __name__ == "__main__":
    main()
