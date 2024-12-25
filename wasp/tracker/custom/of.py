from dataclasses import dataclass, field

import cv2
import numpy as np

XYWH = tuple[int, int, int, int]


def clean_features(features, sigma_threshold=1):
    # TODO: Check why do we need it
    features = features.reshape(-1, 2)
    center = features.mean(axis=0, keepdims=True)
    deviations = np.linalg.norm(features - center, axis=1)

    mean_deviation = deviations.mean()
    std_deviation = deviations.std()

    allowed_deviation = mean_deviation + sigma_threshold * std_deviation
    return features[deviations <= allowed_deviation]


def clean_errors(features, sigma_threshold=0.5):
    center = features.mean(keepdims=True)
    deviations = np.linalg.norm(features - center)

    mean_deviation = deviations.mean()
    std_deviation = deviations.std()

    allowed_deviation = mean_deviation + sigma_threshold * std_deviation
    return features[deviations <= allowed_deviation]


def most_extreme_value(arr):
    return arr[np.argmax(np.abs(arr))]


def extreme_midpoint(data, num_bins=20, threshold=0.05):
    counts, bin_edges = np.histogram(data, bins=num_bins)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    tcounts = threshold * np.max(counts)
    valid_midpoints = midpoints[counts >= tcounts]

    if valid_midpoints.size == 0:
        return np.mean(data)

    return valid_midpoints[np.argmax(np.abs(valid_midpoints))]


def calculate_displacement(old_features, new_features):
    displacement = clean_features(new_features - old_features)
    # plot_histogram(displacement[:0], displacement[:, 1])
    ex = extreme_midpoint(displacement[:, 0])
    ey = extreme_midpoint(displacement[:, 1])

    mx, my = np.mean(displacement, axis=0)
    return mx, my

    return np.mean([ex, mx]), np.mean([ey, my])


def calculate_spread1(x_coords, indices=None):
    if indices is None:
        indices = np.argsort(x_coords)

    x_coords_sorted = x_coords[indices]

    x_differences = np.diff(x_coords_sorted)
    return np.mean(x_differences), indices


def calculate_spread(points):
    return np.max(points) - np.min(points)


def plot_histogram(od, nd):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    print(od)
    print(nd)
    plt.hist(od.ravel(), bins=20, alpha=0.5, label="x")
    plt.hist(nd.ravel(), bins=20, alpha=0.5, label="y")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Old vs New Distances")
    plt.legend()
    plt.show()


def optical_flow(frame1, frame2, features1):
    if features1 is None or features1.size == 0:
        return False, (None, None)

    features2, status, error = cv2.calcOpticalFlowPyrLK(
        frame1,
        frame2,
        features1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.05,
        ),
    )

    if features2 is None or status is None or error is None:
        return False, (features1, None)

    # mean_error = clean_errors(error.squeeze()).mean()
    print(error.mean())

    valid_mask = (status.squeeze() == 1) & (error.squeeze() < np.mean(error))
    features1 = features1[valid_mask]
    features2 = features2[valid_mask]

    return len(features2) > 0, (features1, features2)


def make_sure_includes_all(bbox, features, margin=0.05):
    min_x, min_y = np.min(features, axis=0)
    max_x, max_y = np.max(features, axis=0)

    x, y, w, h = bbox

    new_x = min(min_x, x)
    new_y = min(min_y, y)
    new_w = max(max_x, x + w) - new_x
    new_h = max(max_y, y + h) - new_y

    if new_w > w or new_h > h:
        margin_x = new_w * margin
        margin_y = new_h * margin

        new_x -= margin_x
        new_y -= margin_y
        new_w += 2 * margin_x
        new_h += 2 * margin_y

    return int(new_x), int(new_y), int(new_w), int(new_h)


@dataclass
class OpticalFLowTracker:
    bbox: XYWH = (0, 0, 0, 0)
    last_frame: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    features: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    def init(self, frame, bbox: XYWH) -> None:
        self.last_frame = frame.copy()
        self.bbox = bbox
        self.features = to_features(frame, bbox=self.bbox)
        self.last_features = self.features

    def _calculate(self, frame1, frame2, old_features):
        status, (features1, features2) = optical_flow(
            frame1,
            frame2,
            old_features,
        )
        if not status:
            return status, (features1, features2)

        status, (features2, features1) = optical_flow(
            frame2,
            frame1,
            features2,
        )

        return status, (features1, features2)

    def update(self, frame: np.ndarray) -> tuple[bool, XYWH]:
        status, (old_features, new_features) = self._calculate(
            self.last_frame,
            frame,
            self.features,
        )
        if not status:
            return False, self.bbox

        # Calculate displacement
        dx, dy = calculate_displacement(
            old_features,
            new_features,
        )
        print(dx, dy)

        # Update bounding box position
        x, y, w, h = self.bbox
        x_new = x + dx
        y_new = y + dy

        # Calculate scale change
        # scale_h, scale_w = calculate_scale_change(old_features, new_features)
        h_new = h
        w_new = w

        # Update tracker state
        self.bbox = (int(x_new), int(y_new), int(w_new), int(h_new))
        self.last_features = old_features + (dx, dy)
        self.bbox = make_sure_includes_all(self.bbox, self.last_features)
        self.features = to_features(frame, self.bbox)

        self.last_frame = frame.copy()

        return True, self.bbox

    def plot(self, frame: np.ndarray) -> np.ndarray:
        for feature in self.last_features.reshape(-1, 2):
            x, y = int(feature[0]), int(feature[1])
            frame = cv2.circle(
                frame,
                (x, y),
                radius=2,
                color=(0, 0, 255),
                thickness=-1,
            )
        for feature in self.features.reshape(-1, 2):
            x, y = int(feature[0]), int(feature[1])
            frame = cv2.circle(
                frame,
                (x, y),
                radius=5,
                color=(0, 255, 0),
                thickness=-1,
            )
        return frame


def to_features(
    frame: np.ndarray,
    bbox: XYWH,
) -> np.ndarray:
    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    features = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=1,
        blockSize=1,
    )
    if features is None:
        return np.array([])

    features = features.reshape(-1, 2)
    features[:, 0] += x
    features[:, 1] += y
    return features
