from dataclasses import dataclass
from typing import Callable

import cv2

# from wasp.tracker.boundaries import visualize_features
from wasp.tracker.filter import KalmanFilter
from wasp.tracker.tracker import TemplateMatchingTrackerWithResize


@dataclass
class Segment:
    start_frame: int
    stop_frame: int
    last_frame: int
    bbox: tuple[float, float, float, float]
    tracker: Callable

    def within(self, frame_count):
        return self.start_frame <= frame_count < self.stop_frame


SEGMENTS = {
    "sky": Segment(
        120,
        1000,
        last_frame=580,
        bbox=(1031.0, 721.0, 200.0, 138.0),
        tracker=TemplateMatchingTrackerWithResize,
    ),
    "sky-slimmer": Segment(
        120,
        1000,
        last_frame=580,
        bbox=(1048, 744, 160, 96),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        # tracker=cv2.TrackerCSRT_create,
        # tracker=TemplateMatchingTracker,
        tracker=TemplateMatchingTrackerWithResize,
    ),
    "mixed": Segment(
        580,
        1000,
        last_frame=667,
        # roi=(897, 449, 32, 18),
        bbox=(896, 446, 16, 17),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        # tracker=cv2.TrackerCSRT_create,
        tracker=TemplateMatchingTrackerWithResize,
    ),
    "after-hard-field": Segment(
        734,
        1000,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(536, 333, 49, 36),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        tracker=TemplateMatchingTrackerWithResize,
    ),
    "after-hard-field-b": Segment(
        734,
        1000,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(536, 333, 49, 36),
        # tracker=cv2.legacy.TrackerMOSSE_create,
        tracker=cv2.legacy.TrackerBoosting_create,
    ),
}


def overlay_bbox_on_frame_simple(frame, bbox, max_size=256, o_x=40):
    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    w, h = roi.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_roi = cv2.resize(roi, (new_width, new_height))
    frame_height, frame_width = frame.shape[:2]
    o_y = frame_height - new_height - 10
    frame[o_y : o_y + new_height, o_x : o_x + new_width] = resized_roi


def overlay_template(frame, roi, score, max_size=256, o_x=40):
    w, h = roi.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_roi = cv2.resize(roi, (new_width, new_height))
    cv2.putText(
        resized_roi,
        f"score: {score:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        4,
    )
    frame_height, frame_width = frame.shape[:2]
    o_y = frame_height - new_height - 10
    frame[o_y : o_y + new_height, o_x : o_x + new_width] = resized_roi


def overlay_bbox_on_frame(frame, bbox, max_size=256, o_x=40):
    x, y, w, h = bbox
    center_x, center_y = x + w // 2, y + h // 2
    new_w, new_h = 2 * w, 2 * h
    new_x, new_y = max(0, center_x - new_w // 2), max(0, center_y - new_h // 2)
    roi = frame[
        new_y : new_y + min(frame.shape[0] - new_y, new_h),
        new_x : new_x + min(frame.shape[1] - new_x, new_w),
    ]
    scale = min(max_size / roi.shape[1], max_size / roi.shape[0])
    nroi = cv2.resize(
        roi,
        (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
    )
    o_y = max(10, frame.shape[0] - nroi.shape[0] - 10)
    o_x = min(o_x, frame.shape[1] - nroi.shape[1])
    frame[o_y : o_y + nroi.shape[0], o_x : o_x + nroi.shape[1]] = nroi  # noqa
    return frame


def main():
    cap = cv2.VideoCapture("test.mov")
    segment = SEGMENTS["sky-slimmer"]
    tracker = segment.tracker()
    bbox = segment.bbox
    frame_count = -1
    kalman_filter = None
    # ellipse = None
    prev_frame = None  # Store the previous frame

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            tracker.init(frame, segment.bbox)
            kalman_filter = KalmanFilter(segment.bbox)

        kalman_filter.correct(bbox)
        success, bbox = tracker.update(frame)
        bbox = tuple(map(int, bbox))
        x, y, w, h = bbox

        # Draw the original tracker's bounding box
        # frame, ellipse = visualize_features(frame, roi, ellipse)
        overlay_bbox_on_frame_simple(frame, bbox)
        overlay_template(frame, tracker.template, tracker.max_val, o_x=300)
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
