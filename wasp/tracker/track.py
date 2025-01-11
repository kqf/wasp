import cv2
from matplotlib.patches import draw_bbox

from wasp.tracker.capture import video_dataset
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
    for i, (xframe, label) in enumerate(frames):
        frame = cv2.cvtColor(xframe, cv2.COLOR_BGR2GRAY)
        if tracker is None:
            kfilter = KalmanFilter(label.to_tuple())
            tracker = TemplateMatchingTracker()
            tracker.init(frame, label.to_tuple())

        kfilter.correct(bbox)
        _, bbox = tracker.update(frame)

        draw_bbox(xframe, bbox, (0, 255, 0))
        draw_bbox(xframe, label.to_tuple(), (255, 0, 0))
        # overlay_bbox(xframe, bbox=bbox)
        # tracker.plot(xframe)
        # draw_bbox(xframe, kfilter.predict(), (255, 0, 0))
        cv2.imshow("tracking", xframe)
        if cv2.waitKey() == 27:
            return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
