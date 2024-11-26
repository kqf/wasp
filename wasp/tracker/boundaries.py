import cv2
import numpy as np


def extract_features(image, bounding_box):
    x, y, w, h = bounding_box
    roi = image[y : y + h, x : x + w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(roi_gray, 100, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    largest_contour = max(contours, key=cv2.contourArea)

    points = largest_contour.reshape(-1, 2)
    points[:, 0] += x
    points[:, 1] += y

    return points


def draw_ellipse(image, ellipse, color=(0, 255, 0), thickness=2):
    if not ellipse:
        return image

    center, axes, angle = ellipse

    center = tuple(map(int, center))
    axes = tuple(map(int, axes))

    return cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)


def extract_features_with_ellipse(image, bounding_box, prior_ellipse=None):
    x, y, w, h = bounding_box
    roi = image[y : y + h, x : x + w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi_gray, 100, 200)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    for contour in contours:
        contour[:, 0, 0] += x
        contour[:, 0, 1] += y

    if prior_ellipse:
        # select the countour
        # selected_contour = np.vstack(contours)

        prior_center, prior_axes, _ = prior_ellipse
        prior_major_axis = max(prior_axes)
        selected_contour = min(
            contours,
            key=lambda c: (
                np.linalg.norm(prior_center - np.mean(c[:, 0, :], axis=0))
                if np.linalg.norm(prior_center - np.mean(c[:, 0, :], axis=0))
                <= prior_major_axis
                else float("inf")
            ),
            default=None,
        )
    else:
        bbox_center = np.array([x + w / 2, y + h / 2])
        # selected_contour = min(
        #     contours,
        #     key=lambda c: max(
        #         0,
        #         cv2.pointPolygonTest(c, tuple(bbox_center), True),
        #     ),
        #     default=None,
        # )
        selected_contour = np.vstack(contours)

    ellipse = (
        cv2.fitEllipse(selected_contour)
        if selected_contour is not None and len(selected_contour) >= 5
        else None
    )
    return selected_contour.reshape(-1, 2), ellipse


def extract_features_small(image, bounding_box):
    x, y, w, h = bounding_box
    roi = image[y : y + h, x : x + w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(roi_gray, 50, 150)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    min_area = 10
    max_area = (w - 1) * (h - 1)

    def is_bounding_box_contour(contour, bounding_box):
        x, y, w, h = bounding_box
        x_min, y_min, w_contour, h_contour = cv2.boundingRect(contour)
        return abs(w_contour - w) < 5 and abs(h_contour - h) < 5

    filtered_contours = [
        contour
        for contour in contours
        if min_area < cv2.contourArea(contour) < max_area
        and not is_bounding_box_contour(contour, bounding_box)
    ]

    if filtered_contours:
        largest_contour = max(filtered_contours, key=cv2.contourArea)
    else:
        return np.array([])  # Return empty if no valid contours

    points = largest_contour.reshape(-1, 2)
    if points.size > 0:
        points[:, 0] += x
        points[:, 1] += y

    return points


def draw_features(image, points):
    if points is None:
        return image

    for point in points:
        cv2.circle(
            image,
            (int(point[0]), int(point[1])),
            1,
            (0, 255, 0),
            -1,
        )
    return image


def visualize_features(image, bounding_box, ellipse=None):
    # points = extract_features_small(image, bounding_box)
    points, ellipse = extract_features_with_ellipse(
        image,
        bounding_box,
        ellipse,
    )
    draw_ellipse(image, ellipse)
    return draw_features(image, points), ellipse


samples = {
    "588.png": (863, 432, 16, 17),
    "589.png": (858, 430, 16, 17),
    "590.png": (854, 427, 16, 17),
    "591.png": (850, 424, 16, 17),
}


def overlay_bbox_on_frame(frame, bbox, max_size=256, o_x=40):
    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    scale = min(max_size / w, max_size / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_roi = cv2.resize(roi, (new_width, new_height))
    frame_height, frame_width = frame.shape[:2]
    o_y = frame_height - new_height - 10
    frame[o_y : o_y + new_height, o_x : o_x + new_width] = resized_roi
    return frame


def main():
    # Example usage:
    for sample, bounding_box in samples.items():
        image = cv2.imread(sample)
        points = extract_features_small(image, bounding_box)
        draw_features(image, points)

        overlay_bbox_on_frame(image, bounding_box)

        cv2.imshow("Feature Points", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
