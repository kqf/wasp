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
    contours = to_contours(image, bounding_box)
    selected_contour = filter_contours_with_ellipse(
        contours, bounding_box, prior_ellipse
    )

    if len(selected_contour) < 5:
        return selected_contour.reshape(-1, 2), None

    ellipse = cv2.fitEllipse(selected_contour)
    return selected_contour.reshape(-1, 2), ellipse


def to_contours(image, bounding_box):
    x, y, w, h = bounding_box
    roi_gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi_gray, 100, 200)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    for contour in contours:
        contour[:, 0, 0] += x
        contour[:, 0, 1] += y
    return contours


def filter_contours_with_ellipse(contours, bounding_box, ellipse):
    selected_contour = np.vstack(contours)

    if ellipse is None:
        return selected_contour

    center, axes, angle = ellipse
    adjusted_center = adjust_ellipse_center(center, bounding_box)

    return np.array(
        [
            point
            for point in selected_contour
            if is_inside_ellipse(point, adjusted_center, axes, angle)
        ]
    )


def adjust_ellipse_center(prior_center, bounding_box):
    x, y, w, h = bounding_box
    bbox_center = np.array([x + w / 2, y + h / 2])
    return prior_center + (bbox_center - prior_center)


def is_inside_ellipse(point, ellipse_center, axes, angle):
    cos_angle, sin_angle = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    dx, dy = (
        point[:, 0].item() - ellipse_center[0],
        point[:, 1].item() - ellipse_center[1],
    )
    x_rot = cos_angle * dx + sin_angle * dy
    y_rot = -sin_angle * dx + cos_angle * dy
    return (x_rot**2 / axes[0] ** 2 + y_rot**2 / axes[1] ** 2) <= 1


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


samples = {
    "588.png": (863, 432, 16, 17),
    "589.png": (858, 430, 16, 17),
    "590.png": (854, 427, 16, 17),
    "591.png": (850, 424, 16, 17),
}


def main():
    # Example usage:
    for sample, bounding_box in samples.items():
        image = cv2.imread(sample)
        points = extract_features_small(image, bounding_box)
        draw_features(image, points)
        cv2.imshow("Feature Points", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
