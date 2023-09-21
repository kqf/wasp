from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime
from skimage.transform import SimilarityTransform

from wasp.infer.detection.nn import nninput


@dataclass
class Face:
    kps: np.ndarray
    embedding: Optional[np.ndarray] = None


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


def estimate_norm(
    keypoints: np.ndarray,
    image_size: int = 112,
    expected_size: int = 112,
    expected_keypoints_shape: tuple[int, int] = (5, 2),
) -> np.ndarray:
    if image_size % 112 != 0:
        raise RuntimeError(
            f"Expected the image size multiple of {expected_size}",
        )

    if keypoints.shape != expected_keypoints_shape:
        raise RuntimeError(
            f"Expected the keypoints of shape {expected_keypoints_shape}",
        )

    ratio = float(image_size) / expected_size
    dst = arcface_dst * ratio
    similarity = SimilarityTransform()
    similarity.estimate(keypoints, dst)
    return similarity.params[0:2, :]


def _read_model(model: str):
    session = onnxruntime.InferenceSession(model)
    iconf = session.get_inputs()[0]
    shape, input_name = iconf.shape, iconf.name
    outputs = session.get_outputs()
    output_names = [out.name for out in outputs]
    *_, h, w = shape
    return session, (h, w), input_name, output_names


class ArcFace:
    def __init__(self, model):
        self.session, self.resolution, self.iname, self.onames = _read_model(
            model,
        )

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        # Crop the image using keypoints
        crop = norm_crop(
            image,
            landmark=keypoints,
            image_size=self.resolution[0],
        )
        # Prepare the input
        blob = nninput(
            crop,
            shape=self.resolution,
            std=127.5,
        )

        # Infer from the network
        features = self.session.run(self.onames, {self.iname: blob})[0]
        return features.flatten()
