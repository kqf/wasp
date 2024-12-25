from dataclasses import dataclass

import cv2
import numpy as np

XYWH = tuple[float, float, float, float]


def clean_features(features, sigma_threshold=0.5):
    # TODO: Check why do we need it
    features = features.reshape(-1, 2)
    center = features.mean(axis=0, keepdims=True)
    deviations = np.linalg.norm(features - center, axis=1)

    mean_deviation = deviations.mean()
    std_deviation = deviations.std()

    allowed_deviation = mean_deviation + sigma_threshold * std_deviation
    return features[deviations <= allowed_deviation]


@dataclass
class OpticalFLowTracker:
    bbox: XYWH = (0.0, 0.0, 0.0, 0.0)
    last_frame: np.ndarray = np.empty((0,))
    features: np.ndarray = np.empty((0,))

    def init(self, frame, bbox: XYWH) -> None:
        self.last_frame = frame.clone()
        self.bbox = bbox
        self.features = to_features(frame, bbox=self.bbox)

    def update(self, frmae: np.ndarray) -> tuple[bool, XYWH]:
        # TODO: This should never happen if we don't clean up features
        if self.features.size == 0:
            return False, self.bbox

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_frame,
            frmae,
            self.features,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        if status is None:
            return False, self.bbox

        # Keep only the successfully tracked features
        features = new_features[status.squeeze() == 1]

        if len(features) == 0:
            return False, self.bbox

        # self.features = clean_features(features)
        self.features = features
        self.bbox = self.box_from_features(frmae, self.features)
        return True, self.bbox

    def box_from_features(self, frame, features):
        h, w = frame.shape[:2]

        # Update the bounding box position based on the new feature positions
        new_x1, new_y1 = np.min(features, axis=0).astype(int).flatten()
        new_x2, new_y2 = np.max(features, axis=0).astype(int).flatten()

        # Convert absolute coordinates back to relative values
        rel_x1, rel_y1 = new_x1 / w, new_y1 / h
        rel_x2, rel_y2 = new_x2 / w, new_y2 / h
        return rel_x1, rel_y1, rel_x2, rel_y2


def to_features(
    frame: np.ndarray,
    bbox: XYWH,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    # Convert from relative to absolute coordinates
    h, w = frame.shape[:2]
    abs_x1, abs_y1 = int(x1 * w), int(y1 * h)
    abs_x2, abs_y2 = int(x2 * w), int(y2 * h)

    roi = frame[abs_y1:abs_y2, abs_x1:abs_x2]
    features = cv2.goodFeaturesToTrack(
        roi, maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    if features is None:
        return np.array([])

    features = features.reshape(-1, 2)
    features[:, 0] += abs_x1
    features[:, 1] += abs_y1
    return features
