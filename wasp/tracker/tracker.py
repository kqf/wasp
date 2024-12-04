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


def shift_box(roi, new_w, new_h):
    x, y, w, h = roi

    # Calculate the center of the current bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the new top-left corner based on new width and height
    new_x = center_x - new_w // 2
    new_y = center_y - new_h // 2

    return (new_x, new_y, new_w, new_h)


def resize_bounding_box(bounding_box, image_shape=None):
    x, y, w, h = bounding_box
    center_x = x + w // 2
    center_y = y + h // 2
    new_w = max(w * 2, 30)
    new_h = max(h * 2, 30)
    new_x = center_x - new_w // 2
    new_y = center_y - new_h // 2

    if image_shape is not None:
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, image_shape[1] - new_x)
        new_h = min(new_h, image_shape[0] - new_y)

    return new_x, new_y, new_w, new_h


def extract_features(image, bounding_box):
    x, y, w, h = resize_bounding_box(bounding_box, image.shape)
    roi = image[y : y + h, x : x + w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(roi_gray, 100, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None

    largest_contour = np.concatenate(contours)

    points = largest_contour.reshape(-1, 2)
    points[:, 0] += x
    points[:, 1] += y

    return points


def max_bounding_box(points):
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    return min_x, min_y, width, height


class TemplateMatchingTrackerWithResize:
    def __init__(self, n=3, alpha=0.95, confidence_threshold=0.0):
        self.initialized = False
        self.n = n
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.w = None
        self.h = None
        self.last_position = None
        self.template = None
        self.bkg = cv2.createBackgroundSubtractorMOG2()

    def init(self, frame, roi):
        x, y, w, h = map(int, roi)
        self.template = frame[y : y + h, x : x + w].astype(np.float32)
        self.w = w
        self.h = h
        self.last_position = (x, y, w, h)
        self.initialized = True
        roi_frame = frame[y : y + h, x : x + w]
        self.bkg.apply(roi_frame)

    def update_bbox_with_contours(self, frame, roi):
        points = extract_features(frame, roi)
        x, y, w, h = roi
        if points is None:
            return w, h

        roi_frame = frame[y : y + h, x : x + w]
        for point in points:
            cv2.circle(
                roi_frame,
                (int(point[0] - x), int(point[1] - y)),
                1,
                (0, 255, 0),
                -1,
            )

        nx, ny, new_w, new_h = cv2.boundingRect(points)
        return new_w * 2, new_h * 2

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
            search_area, self.template.astype(np.uint8), cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.confidence_threshold:
            return False, self.last_position

        top_left = (search_x + max_loc[0], search_y + max_loc[1])
        x, y = top_left

        nt = frame[y : y + self.h, x : x + self.w]
        nt = nt.astype(np.float32)
        self.template = self.alpha * self.template + (1 - self.alpha) * nt

        new_w, new_h = self.update_bbox_with_contours(frame, (x, y, w, h))
        print(new_w, new_h)
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


class TemplateMatchingScaled(TemplateMatchingTracker):
    def __init__(self, n=3, scale_factors=None):
        super().__init__(n=n)
        self.scale_factors = scale_factors or [0.8, 0.9, 1.0, 1.1, 1.2]

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

        best_match_val = -1
        best_match_pos = None
        best_scale = 1.0

        for scale in self.scale_factors:
            scaled_template = cv2.resize(
                self.template,
                (
                    int(self.template_width * scale),
                    int(self.template_height * scale),
                ),
            )
            result = cv2.matchTemplate(
                search_area,
                scaled_template,
                cv2.TM_CCOEFF_NORMED,
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_pos = max_loc
                best_scale = scale

        if best_match_pos is None:
            return False, (x, y, w, h)

        top_left = (search_x + best_match_pos[0], search_y + best_match_pos[1])
        x, y = top_left
        w = int(self.template_width * best_scale)
        h = int(self.template_height * best_scale)

        self.last_position = (x, y, w, h)
        self.template = cv2.resize(
            frame[y : y + h, x : x + w],
            (self.template_width, self.template_height),
        )
        return True, (x, y, w, h)
