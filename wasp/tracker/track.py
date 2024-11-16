from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class Segment:
    start_frame: int
    stop_frame: int
    last_frame: int
    roi: tuple[float, float, float, float]
    tracker: Callable

    def within(self, frame_count):
        return self.start_frame <= frame_count < self.stop_frame


SEGMENTS = {
    "sky": Segment(
        120,
        1000,
        last_frame=580,
        roi=(1031.0, 721.0, 200.0, 138.0),
        tracker=cv2.legacy.TrackerMOSSE_create,
    ),
    "sky-slimmer": Segment(
        120,
        1000,
        last_frame=580,
        roi=(1048, 744, 160, 96),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        tracker=cv2.TrackerCSRT_create,
    ),
    "mixed": Segment(
        580,
        1000,
        last_frame=667,
        # roi=(897, 449, 32, 18),
        roi=(890, 435, 29, 36),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        tracker=cv2.TrackerCSRT_create,
    ),
    "after-hard-field": Segment(
        734,
        1000,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        roi=(536, 333, 49, 36),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        tracker=cv2.TrackerCSRT_create,
    ),
    "after-hard-field-b": Segment(
        734,
        1000,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        roi=(536, 333, 49, 36),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        tracker=cv2.legacy.TrackerBoosting_create,
    ),
}


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
        x, y, w, h = initial_roi
        self.w, self.h = w, h
        self.prev_x, self.prev_y = x, y

        # Initialize the Kalman Filter with the initial measurement
        self.correct(x, y)

    def predict(self):
        predicted = self.kf.predict()
        x = int(predicted[0]) - self.w // 2
        y = int(predicted[1]) - self.h // 2
        return x, y, self.w, self.h

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

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


def main():
    cap = cv2.VideoCapture("test.mov")
    segment = SEGMENTS["sky"]
    tracker = segment.tracker()
    roi = segment.roi
    frame_count = -1

    # Initialize Kalman Filter with the initial ROI
    # kalman_filter = KalmanFilter(roi)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            # print(cv2.selectROI("select the area", frame))
            tracker.init(frame, segment.roi)

        success, roi = tracker.update(frame)
        x, y, w, h = map(int, roi)

        # Draw the original tracker's bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if not success:
            cv2.putText(
                frame,
                "Tracking failed",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        # Use Kalman Filter to smooth and validate the tracker's output
        # kx, ky, kw, kh = kalman_filter.smooth_and_validate(roi)

        # Draw the Kalman Filter's smoothed bounding box
        # cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (255, 0, 0), 2)

        cv2.imshow("Object Tracking", frame)
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
