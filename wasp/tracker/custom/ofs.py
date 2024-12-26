from typing import Tuple

import cv2
import numpy as np

XYWH = tuple[int, int, int, int]


class OpticalFLowSimplified:
    def __init__(self):
        self.bbox = None
        self.previous_frame = None
        self.previous_points = np.array([], dtype=np.float32).reshape(0, 2)
        self.object_lost = False

    def init(self, frame: np.ndarray, bbox: XYWH) -> bool:
        self.bbox = bbox
        self.previous_frame = frame.copy()
        mask = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

        self.previous_points = cv2.goodFeaturesToTrack(mask, 100, 0.3, 7)
        if self.previous_points is None:
            self.previous_points = np.array([], dtype=np.float32).reshape(0, 2)
            self.object_lost = True
            return False

        self.previous_points = self.previous_points.reshape(-1, 2)
        self.previous_points[:, 0] += bbox[0]
        self.previous_points[:, 1] += bbox[1]
        self.object_lost = False
        return True

    def displacement(self, frame: np.ndarray) -> Tuple[float, float]:
        if len(self.previous_points) == 0:
            return 0.0, 0.0

        mean_point = np.mean(self.previous_points, axis=0)
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
            self.previous_points,
            None,
        )

        # Filter points based on status
        valid_new_points = new_points[status.flatten() == 1]
        valid_old_points = self.previous_points[status.flatten() == 1]

        if len(valid_new_points) < len(self.previous_points) * 0.5:
            print("Error: Lost too many features, object is lost")
            self.object_lost = True
            return False, self.bbox

        self.previous_frame = frame.copy()
        self.previous_points = valid_new_points

        dx, dy = np.mean(valid_new_points - valid_old_points, axis=0)
        x, y, w, h = self.bbox
        self.bbox = (int(x + dx), int(y + dy), w, h)

        return True, self.bbox
