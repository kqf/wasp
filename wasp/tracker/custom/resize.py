import cv2


def resize_frame(frame, bbox, max_resolution):
    max_w, max_h = max_resolution
    h, w = frame.shape[:2]

    if w <= max_w and h <= max_h:
        return frame, bbox, 1.0  # No resizing needed

    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))

    x, y, bw, bh = bbox
    resized_bbox = (
        int(x * scale),
        int(y * scale),
        int(bw * scale),
        int(bh * scale),
    )

    return resized_frame, resized_bbox, scale


def restore_bbox(bbox, scale):
    if abs(scale - 1.0) < 1e-6:
        return bbox  # No scaling needed
    x, y, w, h = bbox
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))


class ResizedTracker:
    def __init__(
        self, tracker: cv2.Tracker, max_resolution: tuple = (640, 480)
    ):
        self.tracker = tracker
        self.max_resolution = max_resolution
        self.scale = 1.0
        self.last_bbox = None

    def init(self, frame, bbox):
        self.last_bbox = bbox
        resized_frame, resized_bbox, scale = resize_frame(
            frame, bbox, self.max_resolution
        )
        self.scale = scale
        self.tracker.init(resized_frame, resized_bbox)

    def update(self, frame):
        resized_frame, _, scale = resize_frame(
            frame, self.last_bbox, self.max_resolution
        )
        success, new_bbox = self.tracker.update(resized_frame)

        if not success:
            return False, None

        self.last_bbox = restore_bbox(new_bbox, scale)
        return success, self.last_bbox
