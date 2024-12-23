import cv2
import numpy as np


def shift_box(bbox, new_w, new_h):
    x, y, w, h = bbox

    # Calculate the center of the current bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the new top-left corner based on new width and height
    new_x = center_x - new_w // 2
    new_y = center_y - new_h // 2

    return (new_x, new_y, new_w, new_h)


def recalculate_size(
    scores,
    w,
    h,
    threshold=0.01,
):
    height, width = scores.shape
    if height == 0 or width == 0:
        return 2 * w, 2 * h

    lmean = np.mean(scores[:, 0]) - np.max(scores)
    rmean = np.mean(scores[:, -1]) - np.max(scores)
    tmean = np.mean(scores[0, :]) - np.max(scores)
    bmean = np.mean(scores[-1, :]) - np.max(scores)

    resized = cv2.resize(scores, (640, 480))
    cv2.putText(
        resized,
        f"score: {tmean:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        4,
    )
    cv2.imshow("lol1", resized)
    cv2.waitKey()

    w_scale, h_scale = 0.96, 0.96
    if lmean > threshold or rmean > threshold:
        w_scale = 1.2

    if tmean > threshold or bmean > threshold:
        h_scale = 1.2

    return int(w * w_scale), int(h * h_scale)


class TemplateMatchingTrackerWithResize:
    def __init__(self, n=3, alpha=0.9, confidence_threshold=0.0):
        self.initialized = False
        self.n = n
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.w = None
        self.h = None
        self.last_position = None
        self.template = None
        self.max_val = 0

    def init(self, frame, roi):
        x, y, w, h = map(int, roi)
        self.template = frame[y : y + h, x : x + w].astype(np.float32)
        self.w = w
        self.h = h
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
        result = cv2.matchTemplate(
            search_area,
            self.template.astype(np.uint8),
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self.max_val = max_val

        r_x, r_y = max_loc[0] - w // 2, max_loc[1] - h // 2
        roi_template = result[r_y : r_y + h, r_x : r_x + w]
        new_w, new_h = recalculate_size(roi_template, w, h)

        if max_val < self.confidence_threshold:
            return False, self.last_position

        top_left = (search_x + max_loc[0], search_y + max_loc[1])
        x, y = top_left

        nt = frame[y : y + self.h, x : x + self.w]
        nt = nt.astype(np.float32)
        self.template = self.alpha * self.template + (1 - self.alpha) * nt

        # new_w, new_h = recalculate_object_size(frame, (x, y, w, h))
        # print(new_w, new_h)
        # new_w, new_h = self.w, self.h
        beta = 0.8
        gama = 0.8
        self.last_position = (x, y, self.w, self.h)
        self.w = max(int(beta * self.w + (1 - beta) * new_w), 15)
        self.h = max(int(gama * self.h + (1 - gama) * new_h), 15)
        self.last_position = shift_box(self.last_position, self.w, self.h)
        self.template = cv2.resize(
            self.template,
            (self.w, self.h),
        )

        return True, self.last_position
