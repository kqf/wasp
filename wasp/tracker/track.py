import cv2

from wasp.tracker.capture import video_dataset
from wasp.tracker.custom.plot import OverlayTracker, draw_bbox
from wasp.tracker.custom.tm import TemplateMatchingTracker
from wasp.tracker.filter import KalmanFilter
from wasp.tracker.segments import load_segments


def main():
    segment = load_segments("wasp/tracker/segments.json")["sky"]
    frames = video_dataset(
        aname="test-annotations.json",
        iname="test.mov",
        start=segment.start_frame,
        final=segment.stop_frame,
    )
    bbox = segment.bbox
    tracker = None
    kfilter = None
    for i, (frame, label) in enumerate(frames):
        if tracker is None:
            kfilter = KalmanFilter(label.to_tuple())
            tracker = OverlayTracker(TemplateMatchingTracker())
            tracker.init(frame, label.to_tuple())

        kfilter.correct(bbox)
        _, bbox = tracker.update(frame)

        draw_bbox(frame, bbox, (0, 255, 0))
        draw_bbox(frame, label.to_tuple(), (255, 0, 0))
        draw_bbox(frame, kfilter.predict(), (255, 0, 0))
        cv2.imshow("tracking", frame)
        if cv2.waitKey() == 27:
            return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
