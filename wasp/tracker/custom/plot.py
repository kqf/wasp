import cv2


class OverlayTracker:
    def __init__(self, tracker):
        self.tracker = tracker

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def update(self, frame):
        status, bbox = self.tracker.update(frame)
        ibox = list(map(int, bbox))
        overlay_bbox(frame, ibox)
        return status, ibox


class PlotInternalTracker:
    def __init__(self, tracker):
        if not hasattr(tracker, "plot"):
            raise RuntimeError("This tracker doen't implement plot method")
        self.tracker = tracker

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def update(self, frame):
        status, bbox = self.tracker.update(frame)
        self.tracker.plot(frame)
        return status, bbox


def overlay_bbox(frame, bbox, max_size=256, o_x=40, o_y=10):
    x, y, w, h = bbox
    if w == 0 or h == 0:
        return frame
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
