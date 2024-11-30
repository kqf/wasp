from dataclasses import dataclass
from typing import Callable

import cv2

from wasp.tracker.boundaries import visualize_features
from wasp.tracker.filter import KalmanFilter
from wasp.tracker.tracker import TemplateMatchingTracker


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
        # tracker=cv2.TrackerCSRT_create,
        tracker=TemplateMatchingTracker,
    ),
    "mixed": Segment(
        580,
        1000,
        last_frame=667,
        # roi=(897, 449, 32, 18),
        roi=(896, 446, 16, 17),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        # tracker=cv2.TrackerCSRT_create,
        tracker=TemplateMatchingTracker,
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


def overlay_bbox_on_frame(frame, bbox, max_size=256, o_x=40):
    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    scale = min(max_size / w, max_size / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_roi = cv2.resize(roi, (new_width, new_height))
    frame_height, frame_width = frame.shape[:2]
    o_y = frame_height - new_height - 10
    frame[o_y : o_y + new_height, o_x : o_x + new_width] = resized_roi
    return frame


def main():
    cap = cv2.VideoCapture("test.mov")
    segment = SEGMENTS["mixed"]
    tracker = segment.tracker()
    roi = segment.roi
    frame_count = -1
    kalman_filter = None
    ellipse = None
    prev_frame = None  # Store the previous frame

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            tracker.init(frame, segment.roi)
            kalman_filter = KalmanFilter(segment.roi)

        kalman_filter.correct(roi)
        success, roi = tracker.update(frame)
        roi = tuple(map(int, roi))
        x, y, w, h = map(int, roi)

        # Draw the original tracker's bounding box
        frame, ellipse = visualize_features(frame, roi, ellipse)
        overlay_bbox_on_frame(frame, roi)
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

        kx, ky, kw, kh = kalman_filter.predict()
        cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (255, 0, 0), 2)

        if prev_frame is None:
            prev_frame = frame

        combined_frame = cv2.hconcat([prev_frame, frame])
        cv2.imshow("tracking", combined_frame)
        cv2.waitKey()
        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
