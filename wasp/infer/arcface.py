from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime

from wasp.infer.distance import norm_crop
from wasp.infer.nn import nninput


@dataclass
class Face:
    kps: np.ndarray
    embedding: Optional[np.ndarray] = None


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
        crop, _ = norm_crop(
            image,
            keypoints=keypoints,
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
