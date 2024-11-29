from dataclasses import dataclass
from typing import Callable

import cv2

from wasp.tracker.boundaries import visualize_features
from wasp.tracker.filter import KalmanFilter


@dataclass
class Segment:
    start_frame: int
    stop_frame: int
    last_frame: int
    roi: tuple[float, float, float, float]
    tracker: Callable

    def within(self, frame_count):
        return self.start_frame <= frame_count < self.stop_frame


class TemplateMatchingTracker:
    def __init__(self, n=3):
        self.initialized = False
        self.n = n
        self.corners = []  # Class variable to store corners

    def init(self, frame, roi):
        x, y, w, h = map(int, roi)
        self.template = frame[y : y + h, x : x + w].copy()
        self.template_width = w
        self.template_height = h
        self.last_position = (x, y, w, h)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call `init` first.")

        x, y, w, h = self.last_position
        search_width = w * self.n
        search_height = h * self.n

        frame_height, frame_width = frame.shape[:2]
        search_x = max(0, x - (search_width - w) // 2)
        search_y = max(0, y - (search_height - h) // 2)
        search_x_end = min(frame_width, search_x + search_width)
        search_y_end = min(frame_height, search_y + search_height)

        search_area = frame[search_y:search_y_end, search_x:search_x_end]

        result = cv2.matchTemplate(
            search_area,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Adjust the top-left corner of the match to the frame coordinates
        top_left = (search_x + max_loc[0], search_y + max_loc[1])
        x, y = top_left
        w, h = self.template_width, self.template_height

        self.last_position = (x, y, w, h)

        # Define the corners of the bounding box
        self.corners = [
            (x, y),  # top_left
            (x + w, y),  # top_right
            (x, y + h),  # bottom_left
            (x + w, y + h),  # bottom_right
        ]

        return True, (x, y, w, h)


class TemplateMatchingScaled(TemplateMatchingTracker):
    def __init__(self, n=3, scale_factors=None):
        super().__init__(n=n)
        self.scale_factors = (
            scale_factors if scale_factors else [0.8, 0.9, 1.0, 1.1, 1.2]
        )

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call `init` first.")

        x, y, w, h = self.last_position
        search_width = w * self.n
        search_height = h * self.n

        frame_height, frame_width = frame.shape[:2]
        search_x = max(0, x - (search_width - w) // 2)
        search_y = max(0, y - (search_height - h) // 2)
        search_x_end = min(frame_width, search_x + search_width)
        search_y_end = min(frame_height, search_y + search_height)

        search_area = frame[search_y:search_y_end, search_x:search_x_end]

        best_match_val = -1
        best_match_pos = None
        best_scale = 1.0

        for scale in self.scale_factors:
            scaled_template = cv2.resize(
                self.template,
                (
                    int(self.template_width * scale),
                    int(self.template_height * scale),
                ),
            )
            result = cv2.matchTemplate(
                search_area, scaled_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_pos = max_loc
                best_scale = scale

        if best_match_pos is None:
            return False, (x, y, w, h)

        top_left = (search_x + best_match_pos[0], search_y + best_match_pos[1])
        x, y = top_left
        w = int(self.template_width * best_scale)
        h = int(self.template_height * best_scale)

        self.last_position = (x, y, w, h)
        self.template = cv2.resize(
            frame[y : y + h, x : x + w],
            (self.template_width, self.template_height),
        )
        self.corners = [
            (x, y),  # top_left
            (x + w, y),  # top_right
            (x, y + h),  # bottom_left
            (x + w, y + h),  # bottom_right
        ]

        return True, (x, y, w, h)


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
        tracker=TemplateMatchingScaled,
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
    segment = SEGMENTS["sky"]
    tracker = segment.tracker()
    roi = segment.roi
    frame_count = -1
    kalman_filter = None
    ellipse = None

    # Initialize Kalman Filter with the initial ROI

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            # print(cv2.selectROI("select the area", frame))
            tracker.init(frame, segment.roi)
            kalman_filter = KalmanFilter(segment.roi)

        kalman_filter.correct(roi)
        success, roi = tracker.update(frame)
        roi = tuple(map(int, roi))
        x, y, w, h = map(int, roi)
        # success = True
        # if 588 <= frame_count <= 591:
        #     cv2.imwrite(f"{frame_count}.png", frame)
        # # print(frame_count, roi)

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

        # tracker.draw_corners(frame)
        frame, ellipse = visualize_features(frame, roi, ellipse)
        overlay_bbox_on_frame(frame, roi)

        # Use Kalman Filter to smooth and validate the tracker's output
        kx, ky, kw, kh = kalman_filter.predict()
        # Draw the Kalman Filter's smoothed bounding box
        cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (255, 0, 0), 2)
        cv2.imshow("Object Tracking", frame)
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
