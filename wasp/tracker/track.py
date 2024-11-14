from dataclasses import dataclass
from typing import Callable

import cv2


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
}


def main():
    cap = cv2.VideoCapture("test.mov")
    segment = SEGMENTS["sky"]
    tracker = segment.tracker()
    roi = segment.roi
    frame_count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"Current frame count {frame_count}", roi)

        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            tracker.init(frame, segment.roi)

        success, roi = tracker.update(frame)
        # success = False

        # if not success:
        # tracker.init(frame, roi_old)

        if success:
            (x, y, w, h) = tuple(map(int, roi))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(
                frame,
                "Tracking failed",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Object Tracking", frame)
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
