from pathlib import Path

import cv2
import numpy as np


def dump_database(output: str, descriptors) -> None:
    np.savez_compressed(output, descriptors=descriptors)


def create_feature_database(impath: str) -> list[np.ndarray]:
    path = Path(impath)
    sift = cv2.SIFT_create()

    db = []
    for file in path.glob("*"):
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            db.append(descriptors)
    return db
