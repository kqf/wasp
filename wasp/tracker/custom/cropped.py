import cv2


def crop_region(frame, bbox, margin):
    x, y, w, h = map(int, bbox)

    x1, y1 = x - margin, y - margin
    x2, y2 = x + w + margin, y + h + margin

    if x1 < 0:
        x2 = min(frame.shape[1], x2 + abs(x1))  # Shift crop right
        x1 = 0
    if y1 < 0:
        y2 = min(frame.shape[0], y2 + abs(y1))  # Shift crop down
        y1 = 0
    if x2 > frame.shape[1]:
        x1 = max(0, x1 - (x2 - frame.shape[1]))  # Shift crop left
        x2 = frame.shape[1]
    if y2 > frame.shape[0]:
        y1 = max(0, y1 - (y2 - frame.shape[0]))  # Shift crop up
        y2 = frame.shape[0]

    cropped = frame[y1:y2, x1:x2]
    new_bbox = (x - x1, y - y1, w, h)
    return cropped, new_bbox, x1, y1


def map_to_original(crop_x, crop_y, new_bbox):
    cx, cy, w, h = map(int, new_bbox)
    return (crop_x + cx, crop_y + cy, w, h)


class CroppedTracker:
    def __init__(self, tracker: cv2.Tracker, pad: int = 1):
        self.tracker = tracker
        self.pad = pad
        self.last_bbox = None

    def init(self, frame, bbox):
        self.last_bbox = bbox
        *_, w, _ = self.last_bbox

        cropped, new_bbox, crop_x, crop_y = crop_region(
            frame, bbox, self.pad * w
        )
        self.tracker.init(cropped, new_bbox)
        self.crop_x, self.crop_y = crop_x, crop_y  # Store cropping offsets

    def update(self, frame):
        *_, w, _ = self.last_bbox
        cropped, _, crop_x, crop_y = crop_region(
            frame, self.last_bbox, self.pad * w
        )

        success, new_bbox = self.tracker.update(cropped)
        if not success:
            return False, None
        self.last_bbox = map_to_original(crop_x, crop_y, new_bbox)
        return success, self.last_bbox
