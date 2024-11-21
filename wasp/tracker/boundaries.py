import cv2
import numpy as np


def extract_features(image, bounding_box):
    x, y, w, h = bounding_box
    roi = image[y : y + h, x : x + w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(roi_gray, 100, 200)

    corners = cv2.goodFeaturesToTrack(
        edges, maxCorners=100, qualityLevel=0.01, minDistance=10
    )

    if corners is not None:
        corners = corners.reshape(-1, 2)
        corners[:, 0] += x
        corners[:, 1] += y

    return corners


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

    largest_contour = max(contours, key=cv2.contourArea)
    # convex_hull = cv2.convexHull(largest_contour)
    points = largest_contour.reshape(-1, 2)

    # points = []
    # for contour in contours:
    #     points.extend(point[0] for point in contour)

    points = np.array(points)
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


def visualize_features(image, bounding_box):
    points = extract_features_small(image, bounding_box)
    return draw_features(image, points)


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
