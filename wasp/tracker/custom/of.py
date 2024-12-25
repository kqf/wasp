from dataclasses import dataclass, field

import cv2
import numpy as np

XYWH = tuple[int, int, int, int]


def clean_features(features, sigma_threshold=0.5):
    # TODO: Check why do we need it
    features = features.reshape(-1, 2)
    center = features.mean(axis=0, keepdims=True)
    deviations = np.linalg.norm(features - center, axis=1)

    mean_deviation = deviations.mean()
    std_deviation = deviations.std()

    allowed_deviation = mean_deviation + sigma_threshold * std_deviation
    return features[deviations <= allowed_deviation]


def most_extreme_value(arr):
    return arr[np.argmax(np.abs(arr))]


def calculate_displacement(old_features, new_features):
    displacement = new_features - old_features
    # plot_histogram(displacement[0], displacement[1])
    ex = most_extreme_value(displacement[:, 0])
    ey = most_extreme_value(displacement[:, 1])

    mx, my = np.mean(displacement, axis=0)

    return np.mean([ex, mx]), np.mean([ey, my])


def calculate_spread1(x_coords, indices=None):
    if indices is None:
        indices = np.argsort(x_coords)

    x_coords_sorted = x_coords[indices]

    x_differences = np.diff(x_coords_sorted)
    return np.mean(x_differences), indices


def calculate_spread(points):
    return np.max(points) - np.min(points)


def calculate_scale_change(of, nf):
    od_x = calculate_spread(of[:, 0])
    od_y = calculate_spread(of[:, 1])

    nd_x = calculate_spread(nf[:, 0])
    nd_y = calculate_spread(nf[:, 1])

    # plot_histogram(od_x, nd_y)
    scale_x = (nd_x) / (od_x + 1e-8) if od_x > 0.0001 else 1.0
    scale_y = (nd_y) / (od_y + 1e-8) if od_y > 0.0001 else 1.0

    print("scale", scale_x, scale_y, nd_x, od_x)
    return scale_x, scale_y


def plot_histogram(od, nd):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(od.ravel(), bins=100, alpha=0.5, label="Old Distances")
    plt.hist(nd.ravel(), bins=100, alpha=0.5, label="New Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Old vs New Distances")
    plt.legend()
    plt.show()


@dataclass
class OpticalFLowTracker:
    bbox: XYWH = (0, 0, 0, 0)
    last_frame: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    features: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    def init(self, frame, bbox: XYWH) -> None:
        self.last_frame = frame.copy()
        self.bbox = bbox
        self.features = to_features(frame, bbox=self.bbox)

    def _calculate(self, frame1, frame2, old_features):
        if old_features.size == 0:
            return False, self.bbox

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            frame1,
            frame2,
            old_features,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        if status is None:
            return False, (old_features, new_features)

        new_features = new_features[status.squeeze() == 1]
        old_features = old_features[status.squeeze() == 1]

        return len(new_features) != 0, (old_features, new_features)

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
        self.features = to_features(frame, self.bbox)
        self.last_frame = frame.copy()

        return True, self.bbox

    def plot(self, frame: np.ndarray) -> np.ndarray:
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
        blockSize=4,
    )
    if features is None:
        return np.array([])

    features = features.reshape(-1, 2)
    features[:, 0] += x
    features[:, 1] += y
    return features
