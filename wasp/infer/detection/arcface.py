from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime
from skimage import transform as trans

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


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]


class ArcFace:
    def __init__(self, model_file):
        self.session = onnxruntime.InferenceSession(model_file)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        *_, h, w = input_shape
        self.input_size = (h, w)
        outputs = self.session.get_outputs()
        output_names = [out.name for out in outputs]
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        # Crop the image using keypoints
        crop = norm_crop(
            image,
            landmark=keypoints,
            image_size=self.input_size[0],
        )
        # Prepare the input
        blob = nninput(
            crop,
            shape=self.input_size,
            std=127.5,
        )

        # Infer from the network
        features = self.session.run(
            self.output_names,
            {self.input_name: blob},
        )[0]

        return features.flatten()
