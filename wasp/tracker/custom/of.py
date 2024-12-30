from dataclasses import dataclass, field

import cv2
import numpy as np

XYWH = tuple[int, int, int, int]


def clean_features(features, sigma_threshold=1):
    features = features.reshape(-1, 2)
    center = features.mean(axis=0, keepdims=True)
    deviations = np.linalg.norm(features - center, axis=1)

    mean_deviation = deviations.mean()
    std_deviation = deviations.std()

    allowed_deviation = mean_deviation + sigma_threshold * std_deviation
    return features[deviations <= allowed_deviation]


def calculate_displacement(old_features, new_features):
    displacement = clean_features(new_features - old_features)
    return np.mean(displacement, axis=0)


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
        winSize=(17, 17),
        maxLevel=5,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            100,
            0.05,
        ),
    )

    if features2 is None or status is None or error is None:
        return False, (features1, None)

    mean_error = np.mean(error)
    valid_mask = (status.squeeze() == 1) & (error.squeeze() < mean_error)
    features1 = features1[valid_mask]
    features2 = features2[valid_mask]

    return len(features2) > 0, (features1, features2)


def features_to_box(feaures) -> XYWH:
    min_x, min_y = np.min(feaures, axis=0)
    max_x, max_y = np.max(feaures, axis=0)

    curr_x = min_x
    curr_y = min_y
    curr_w = max_x - min_x
    curr_h = max_y - min_y
    return int(curr_x), int(curr_y), int(curr_w), int(curr_h)


def add_bbox(a, b, weight) -> XYWH:
    curr_x, curr_y, curr_w, curr_h = b
    new_x, new_y, new_w, new_h = a
    new_x = (1 - weight) * new_x + weight * curr_x
    new_y = (1 - weight) * new_y + weight * curr_y
    new_w = (1 - weight) * new_w + weight * curr_w
    new_h = (1 - weight) * new_h + weight * curr_h
    return int(new_x), int(new_y), int(new_w), int(new_h)


def union_box(bbox1: XYWH, bbox2: XYWH) -> XYWH:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the top-left corner of the union
    union_x = min(x1, x2)
    union_y = min(y1, y2)

    # Calculate the bottom-right corner of the union
    union_x_max = max(x1 + w1, x2 + w2)
    union_y_max = max(y1 + h1, y2 + h2)

    # Calculate width and height of the union
    union_w = union_x_max - union_x
    union_h = union_y_max - union_y

    return union_x, union_y, union_w, union_h


def increase_by_margin(bbox: XYWH, margin: float) -> XYWH:
    new_x, new_y, new_w, new_h = bbox
    margin_x = new_w * margin
    margin_y = new_h * margin

    x = new_x - margin_x
    y = new_y - margin_y
    w = new_w + 2 * margin_x
    h = new_h + 2 * margin_y
    return int(x), int(y), int(w), int(h)


def make_sure_includes_all(
    bbox,
    features,
    current_features,
    margin=0.5,
    weight=0.4,
    n=0.5,
) -> XYWH:
    x, y, w, h = bbox

    def remove_outliers(data):
        data[:, 0] = np.clip(data[:, 0], x - n * w, x + (1 + n) * w)
        data[:, 1] = np.clip(data[:, 1], y - n * h, y + (1 + n) * h)
        return data

    features = remove_outliers(features)
    fbox = features_to_box(features)
    nbox = union_box(bbox, fbox)
    add_bbox(bbox, nbox, weight)
    # If the bounding box needs to expand, add the margin
    if nbox == bbox:
        nbox = features_to_box(current_features)
        nbox = increase_by_margin(nbox, margin=margin)
        nbox = add_bbox(bbox, nbox, weight)
    return nbox


def calculate_scale_change(old_features, new_features):
    old_center = np.mean(old_features, axis=0)
    old_distances = np.linalg.norm(old_features - old_center, axis=0)

    new_center = np.mean(new_features, axis=0)
    new_distances = np.linalg.norm(new_features - new_center, axis=0)
    scale_ratio = new_distances / old_distances
    return np.nan_to_num(scale_ratio, nan=1.0)


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
        # Update bounding box position
        _, _, w, h = self.bbox
        x_new, y_new = np.mean(new_features, axis=0)
        x_new -= w // 2
        y_new -= h // 2

        # Calculate scale change
        scale_w, scale_h = calculate_scale_change(old_features, new_features)
        alpha = 0.1
        h_new = alpha * h * scale_h + (1 - alpha) * h
        w_new = alpha * w * scale_w + (1 - alpha) * w

        # Update tracker state
        self.bbox = (int(x_new), int(y_new), int(w_new), int(h_new))
        self.last_features = old_features + (dx, dy)
        # self.bbox = make_sure_includes_all(
        #     self.bbox,
        #     self.last_features,
        #     new_features,
        # )
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
