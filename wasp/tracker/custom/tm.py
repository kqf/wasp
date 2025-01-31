import cv2
import numpy as np


class TemplateMatchingTracker:
    def __init__(self, n=3, alpha=0.5, confidence_threshold=0.0):
        self.initialized = False
        self.n = n
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold

    def init(self, frame, roi):
        x, y, w, h = map(int, roi)
        self.template = frame[y : y + h, x : x + w].astype(
            np.float32
        )  # Store as float for averaging
        self.template_width = w
        self.template_height = h
        self.last_position = (x, y, w, h)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call `init` first.")

        x, y, w, h = self.last_position
        search_width = w * self.n
        search_height = h * self.n

        frame_height, frame_width = frame.shape[:2]
        search_x = max(0, x - (search_width - w) // 2)
        search_y = max(0, y - (search_height - h) // 2)
        search_x_end = min(frame_width, search_x + search_width)
        search_y_end = min(frame_height, search_y + search_height)

        search_area = frame[search_y:search_y_end, search_x:search_x_end]

        # Template matching
        result = cv2.matchTemplate(
            search_area,
            self.template.astype(np.uint8),
            cv2.TM_CCOEFF_NORMED,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val < self.confidence_threshold:
            # Skip update if confidence is low
            return False, self.last_position

        # Adjust the top-left corner of the match to the frame coordinates
        top_left = (search_x + max_loc[0], search_y + max_loc[1])
        x, y = top_left

        self.last_position = (x, y, w, h)

        # Update the template using exponential moving average
        nt = frame[y : y + h, x : x + w].astype(np.float32)
        self.template = self.alpha * self.template + (1 - self.alpha) * nt

        return True, (x, y, w, h)

    def plot(self, frame, max_size=256, o_x=380, score=-0.1):
        h, w = self.template.shape[:2]
        scale = min(max_size / w, max_size / h)
        n_w, n_h = int(w * scale), int(h * scale)
        resized_roi = cv2.resize(self.template, (n_w, n_h))
        cv2.putText(
            resized_roi,
            f"score: {score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            4,
        )
        fh, _ = frame.shape[:2]
        o_y = fh - n_h - 10
        frame[o_y : o_y + n_h, o_x : o_x + n_w] = resized_roi
        return frame
