import cv2


def crop_region(frame, bbox, pad):
    x, y, w, h = map(int, bbox)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    cropped = frame[y1:y2, x1:x2]
    new_bbox = (x - x1, y - y1, w, h)
    return cropped, new_bbox, x1, y1  # Also return the crop offsets


def map_to_original(crop_x, crop_y, new_bbox):
    cx, cy, w, h = map(int, new_bbox)
    return (crop_x + cx, crop_y + cy, w, h)


class CroppedTracker:
    def __init__(self, tracker: cv2.Tracker, pad: int = 100):
        self.tracker = tracker
        self.pad = pad
        self.last_bbox = None

    def init(self, frame, bbox):
        self.last_bbox = bbox
        cropped, new_bbox, crop_x, crop_y = crop_region(frame, bbox, self.pad)
        self.tracker.init(cropped, new_bbox)
        self.crop_x, self.crop_y = crop_x, crop_y  # Store cropping offsets

    def update(self, frame):
        cropped, _, crop_x, crop_y = crop_region(
            frame, self.last_bbox, self.pad
        )
        success, new_bbox = self.tracker.update(cropped)
        if not success:
            return False, None
        self.last_bbox = map_to_original(crop_x, crop_y, new_bbox)
        return success, self.last_bbox
