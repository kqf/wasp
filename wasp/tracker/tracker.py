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


def shift_box(bbox, new_w, new_h):
    x, y, w, h = bbox

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


# def is_valid_contour(contour, roi):
#     x, y, w, h = roi
#     for point in contour:
#         px, py = point[0]
#         if x <= px < x + w and y <= py < y + h:
#             return False
#     return True


def is_valid_contour(contour, roi):
    x, y, w, h = roi
    contour_points = contour.reshape(-1, 2)
    cx = np.mean(contour_points[:, 0])
    cy = np.mean(contour_points[:, 1])

    return not x <= cx < x + w or not y <= cy < y + h


def valid_n(contour, bbox, n=0.9):
    x, y, w, h = bbox
    contour_points = contour.reshape(-1, 2)

    inside_x = (contour_points[:, 0] >= x) & (contour_points[:, 0] < x + w)
    inside_y = (contour_points[:, 1] >= y) & (contour_points[:, 1] < y + h)

    inside_bbox = inside_x & inside_y
    num_points_inside = np.sum(inside_bbox)

    percentage_inside = num_points_inside / contour_points.shape[0]

    return percentage_inside >= n


def within_bbox(contour, bbox):
    x, y, w, h = bbox
    contour_points = contour.reshape(-1, 2)

    inside_x = (contour_points[:, 0] >= x) & (contour_points[:, 0] < x + w)
    inside_y = (contour_points[:, 1] >= y) & (contour_points[:, 1] < y + h)

    inside_bbox = inside_x & inside_y
    return contour[inside_bbox]


def max_bounding_box(points):
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    return min_x, min_y, width, height


def extract_features(image, bbox):
    x, y, w, h = resize_bounding_box(bbox, image.shape)
    roi = image[y : y + h, x : x + w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi_gray, 100, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    for c in contours:
        for points in c:
            points[:, 0] += x
            points[:, 1] += y

    if not contours:
        return None

    all_points = np.concatenate(contours)
    points = all_points.reshape(-1, 2)
    return points


def shift_ellipse_to_the_new_center(ellipse, bbox):
    if ellipse is None:
        return ellipse

    (center, (major_axis, minor_axis), angle) = ellipse
    x, y, w, h = bbox
    new_center = (x + w // 2, y + h // 2)
    return (new_center, (major_axis, minor_axis), angle)


def within_ellipse(points, bbox, ellipse):
    if not ellipse:
        return points
    (center, (major_axis, minor_axis), angle) = ellipse
    cx, cy = center
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    points_shifted = points - np.array([cx, cy])
    rotated_points = np.column_stack(
        (
            points_shifted[:, 0] * cos_angle
            + points_shifted[:, 1] * sin_angle,
            -points_shifted[:, 0] * sin_angle
            + points_shifted[:, 1] * cos_angle,
        )
    )
    a = major_axis / 2
    b = minor_axis / 2
    ellipse_equation = (rotated_points[:, 0] ** 2) / (a**2) + (
        rotated_points[:, 1] ** 2
    ) / (b**2)
    inside_ellipse = ellipse_equation <= 1
    return points[inside_ellipse]


def recalculate_object_size(frame, bbox, ellipse=None):
    x, y, w, h = bbox
    points = extract_features(frame, bbox)
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

    # Could you implement this function
    shifted_ellipse = shift_ellipse_to_the_new_center(ellipse, bbox)

    # And this function
    points = within_ellipse(points, bbox, shifted_ellipse)
    # nx, ny, new_w, new_h = cv2.boundingRect(points)
    if len(points) < 5:
        return w, h, None

    ellipse = cv2.fitEllipse(points)

    if ellipse is None:
        return w, h

    center, (major_axis, minor_axis), angle = ellipse
    center = tuple(map(int, center))
    cv2.ellipse(
        frame,
        center,
        (int(major_axis), int(minor_axis)),
        angle,
        0,
        360,
        (0, 255, 0),
        2,
    )

    new_w = max(major_axis, w)
    new_h = max(minor_axis, h)

    return new_w * 2, new_h * 2, ellipse


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
        *_, self.ellipse = recalculate_object_size(frame, roi)

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

        if max_val < self.confidence_threshold:
            return False, self.last_position

        top_left = (search_x + max_loc[0], search_y + max_loc[1])
        x, y = top_left

        nt = frame[y : y + self.h, x : x + self.w]
        nt = nt.astype(np.float32)
        self.template = self.alpha * self.template + (1 - self.alpha) * nt

        new_w, new_h, self.ellipse = recalculate_object_size(
            frame, (x, y, w, h), self.ellipse
        )
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
