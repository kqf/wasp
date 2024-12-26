from typing import Tuple

import cv2
import numpy as np

XYWH = tuple[int, int, int, int]


class OpticalFLowSimplified:
    def __init__(self):
        self.bbox = None
        self.previous_frame = None
        self.features = np.array([], dtype=np.float32).reshape(0, 2)
        self.object_lost = False

    def init(self, frame: np.ndarray, bbox: XYWH) -> bool:
        self.bbox = bbox
        self.previous_frame = frame.copy()
        mask = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

        self.features = cv2.goodFeaturesToTrack(mask, 100, 0.3, 7)
        if self.features is None:
            self.features = np.array([], dtype=np.float32).reshape(0, 2)
            self.object_lost = True
            return False

        self.features = self.features.reshape(-1, 2)
        self.features[:, 0] += bbox[0]
        self.features[:, 1] += bbox[1]
        self.object_lost = False
        return True

    def displacement(self, frame: np.ndarray) -> Tuple[float, float]:
        if len(self.features) == 0:
            return 0.0, 0.0

        mean_point = np.mean(self.features, axis=0)
        center = np.array([frame.shape[1] / 2.0, frame.shape[0] / 2.0])

        displacement = mean_point - center
        return displacement[0], displacement[1]

    def update(self, frame: np.ndarray) -> Tuple[bool, XYWH]:
        if self.object_lost:
            return False, self.bbox

        if self.previous_frame is None:
            return False, self.bbox

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_frame,
            frame,
            self.features,
            None,
        )

        # Filter points based on status
        valid_new_points = new_points[status.flatten() == 1]
        valid_old_points = self.features[status.flatten() == 1]

        if len(valid_new_points) < len(self.features) * 0.5:
            print("Error: Lost too many features, object is lost")
            self.object_lost = True
            return False, self.bbox

        dx, dy = np.mean(valid_new_points - valid_old_points, axis=0)
        x, y, w, h = self.bbox
        self.bbox = (int(x + dx), int(y + dy), w, h)
        self.last_features = self.features + (dx, dy)
        self.features = valid_new_points
        self.previous_frame = frame.copy()

        return True, self.bbox

    def plot(self, frame: np.ndarray) -> np.ndarray:
        for feature in self.last_features.reshape(-1, 2):
            x, y = int(feature[0]), int(feature[1])
            frame = cv2.circle(
                frame,
                (x, y),
                radius=2,
                color=(0, 0, 255),
                thickness=-1,
            )
        for feature in self.features.reshape(-1, 2):
            x, y = int(feature[0]), int(feature[1])
            frame = cv2.circle(
                frame,
                (x, y),
                radius=5,
                color=(0, 255, 0),
                thickness=-1,
            )
        return frame
