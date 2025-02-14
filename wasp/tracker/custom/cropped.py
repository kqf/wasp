import cv2


def crop_region(frame, bbox, pad):
    x, y, w, h = map(int, bbox)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    cropped = frame[y1:y2, x1:x2]
    new_bbox = (x - x1, y - y1, w, h)
    return cropped, new_bbox


def map_to_original(prev_bbox, new_bbox):
    px, py, _, _ = map(int, prev_bbox)
    cx, cy, w, h = map(int, new_bbox)
    return (px + cx, py + cy, w, h)


class CroppedTracker:
    def __init__(self, tracker: cv2.Tracker, pad: int = 100):
        self.tracker = tracker
        self.pad = pad
        self.last_bbox = None

    def start(self, frame, bbox):
        self.last_bbox = bbox
        cropped, new_bbox = crop_region(frame, bbox, self.pad)
        self.tracker.init(cropped, new_bbox)

    def track(self, frame):
        cropped, _ = crop_region(frame, self.last_bbox, self.pad)
        success, new_bbox = self.tracker.update(cropped)
        if not success:
            return False, None
        self.last_bbox = map_to_original(self.last_bbox, new_bbox)
        return True, self.last_bbox
