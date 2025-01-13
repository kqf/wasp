import cv2


class GrayscaleTracker:
    def __init__(self, tracker):
        self.tracker = tracker

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def update(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        status, bbox = self.tracker.update(grayscale)
        return status, bbox
