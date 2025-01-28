import cv2
import numpy as np


class KalmanFilter:
    def __init__(self, initial_roi):
        self.kf = cv2.KalmanFilter(
            4, 2
        )  # 4 dynamic params (x, y, dx, dy), 2 measured params (x, y)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Extract initial upper-left corner and size from the ROI
        self.correct(initial_roi)

    def predict(self):
        predicted = self.kf.predict()
        x = int(predicted[0]) - self.w // 2
        y = int(predicted[1]) - self.h // 2
        return x, y, self.w, self.h

    def correct(self, roi):
        x, y, w, h = map(int, roi)
        center_x = x + w // 2
        center_y = y + h // 2
        measurement = np.array(
            [
                [np.float32(center_x)],
                [np.float32(center_y)],
            ],
        )
        self.kf.correct(measurement)
        self.w = w
        self.h = h

    def smooth_and_validate(self, roi):
        x, y, w, h = map(int, roi)
        self.w, self.h = w, h  # Update width and height with the latest ROI
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate velocity (change in position)
        velocity_x = center_x - (self.prev_x + self.w // 2)
        velocity_y = center_y - (self.prev_y + self.h // 2)

        # Check for sudden, unrealistic jumps
        if abs(velocity_x) > 50 or abs(velocity_y) > 50:
            # Use Kalman prediction if the jump is too large
            x, y, w, h = self.predict()
        else:
            # Update Kalman Filter with the new position
            self.correct(center_x, center_y)
            x = center_x - w // 2
            y = center_y - h // 2

        # Update previous position
        self.prev_x, self.prev_y = x, y

        return x, y, self.w, self.h
