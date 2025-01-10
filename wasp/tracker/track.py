import cv2

from wasp.tracker.capture import video_dataset
from wasp.tracker.custom.tm import TemplateMatchingTracker
from wasp.tracker.filter import KalmanFilter
from wasp.tracker.segments import load_segments


def overlay_bbox(frame, bbox, max_size=256, o_x=40, o_y=10):
    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    scale = min(max_size / w, max_size / h)
    n_w, n_h = int(w * scale), int(h * scale)
    o_y_ = frame.shape[0] - n_h - o_y
    frame[o_y_ : o_y_ + n_h, o_x : o_x + n_w] = cv2.resize(roi, (n_w, n_h))
    return frame


def draw_bbox(frame, xywh, color=(0, 255, 0)):
    if xywh is None:
        return
    bbox = tuple(map(int, xywh))
    x, y, w, h = bbox
    return cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


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
        overlay_bbox(xframe, bbox=bbox)
        # tracker.plot(xframe)
        # draw_bbox(xframe, kfilter.predict(), (255, 0, 0))
        cv2.imshow("tracking", xframe)
        if cv2.waitKey() == 27:
            return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
