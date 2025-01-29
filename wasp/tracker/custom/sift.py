import cv2
import numpy as np


def max_bounding_box(points):
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    return min_x, min_y, width, height


def add_weighted(a, b, alpha):
    h, w, _ = b.shape
    r = cv2.resize(
        a,
        (w, h),
    )
    return (alpha * r + (1 - alpha) * b).astype(np.uint8)


def add_weighted_(a, b, alpha):
    return alpha * a + (1 - alpha) * b


def calculate_search_area(frame, last_position, n=3):
    x, y, w, h = last_position
    search_w = w * n
    search_h = h * n
    frame_height, frame_width = frame.shape[:2]

    # Calculate search area coordinates
    search_x = max(0, x - (search_w - w) // 2)
    search_y = max(0, y - (search_h - h) // 2)
    search_x_end = min(frame_width, search_x + search_w)
    search_y_end = min(frame_height, search_y + search_h)

    return frame[search_y:search_y_end, search_x:search_x_end], (
        search_x,
        search_y,
        search_w,
        search_h,
    )


def select_good_matches(
    descriptors1,
    descriptors2,
):
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
    return [m for m, n in matches]


def calculate_shift(src_pts, dst_pts):
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)
    dx, dy = dst_centroid - src_centroid
    return dx, dy


def calculate_scale(src_pts, dst_pts):
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)
    src_distances = np.linalg.norm(src_pts - src_centroid, axis=1)
    dst_distances = np.linalg.norm(dst_pts - dst_centroid, axis=1)
    scale = np.median(dst_distances) / max(np.median(src_distances), 1e-5)
    return scale * 0.0 + 1


def ensure_within_frame(frame, new_x, new_y, new_w, new_h):
    frame_height, frame_width = frame.shape[:2]
    new_x = max(0, min(frame_width - 1, new_x))
    new_y = max(0, min(frame_height - 1, new_y))
    new_w = max(1, min(frame_width - new_x, new_w))
    new_h = max(1, min(frame_height - new_y, new_h))
    return new_x, new_y, new_w, new_h


def plot_detected_features(
    frame,
    keypoints,
):
    frame_with_keypoints = cv2.drawKeypoints(
        frame,
        keypoints,
        None,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
    cv2.imshow("template", frame_with_keypoints)


def compare_frames(
    frame1,
    keypoints1,
    frame2,
    keypoints2,
    good_matches,
):
    img_matches = cv2.drawMatches(
        frame1,
        keypoints1,
        frame2,
        keypoints2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("keypts", img_matches)
    cv2.waitKey()
    return img_matches


class SIFTTracker:
    def __init__(self, n=3, confidence_threshold=0.01, min_matches=10):
        self.initialized = False
        self.n = n
        self.confidence_threshold = confidence_threshold
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()

    def init(self, frame, roi):
        x, y, w, h = map(int, roi)
        self.template = frame[y : y + h, x : x + w]
        self.last_position = (x, y, w, h)
        self.initialized = True
        self.prev_frame = frame

    def _detect(self, image, coords) -> tuple:
        kpts, descriptors = self.sift.detectAndCompute(image, None)
        x, y, *_ = coords
        for point in kpts:
            point.pt = point.pt[0] + x, point.pt[1] + y
        return kpts, descriptors

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call `init` first.")

        keypoints_t, descriptors_t = self._detect(
            self.template,
            self.last_position,
        )

        # Calculate the search area
        search_area, sbox = calculate_search_area(
            frame,
            self.last_position,
            self.n,
        )

        # Detect keypoints and descriptors in the search area
        keypoints_frame, descriptors_frame = self._detect(
            search_area,
            sbox,
        )

        if descriptors_frame is None or len(keypoints_frame) == 0:
            print("No keypoints found")
            return False, self.last_position

        # Select good matches
        good_matches = select_good_matches(descriptors_t, descriptors_frame)

        compare_frames(
            self.prev_frame,
            keypoints_t,
            frame,
            keypoints_frame,
            good_matches=good_matches,
        )

        if len(good_matches) < self.min_matches:
            print("No good matches")
            return False, self.last_position

        # Extract source and destination points
        src_pts = np.float32(
            [keypoints_t[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 2)

        dst_pts = np.float32(
            [keypoints_frame[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 2)

        # Calculate the displacement and scale
        dx, dy = calculate_shift(src_pts, dst_pts)
        # scale = calculate_scale(src_pts, dst_pts)
        x, y, w, h = self.last_position
        *_, w1, h1 = max_bounding_box(dst_pts)
        # Update the bounding box position and size
        new_x = int(x + dx)
        new_y = int(y + dy)
        new_w = int(add_weighted_(w, w1, 0.99))  # Scale the width
        new_h = int(add_weighted_(h, h1, 0.99))  # Scale the height

        # Ensure the new bounding box is within frame bounds
        new_x, new_y, new_w, new_h = ensure_within_frame(
            frame, new_x, new_y, new_w, new_h
        )

        # Update template with blending
        new_template = frame[new_y : new_y + new_h, new_x : new_x + new_w]
        # alpha = 0
        # self.template = add_weighted(self.template, new_template, alpha)
        self.template = new_template
        self.last_position = (new_x, new_y, new_w, new_h)
        self.prev_frame = frame

        return True, self.last_position
