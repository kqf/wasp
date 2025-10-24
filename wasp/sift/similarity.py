import numpy as np

from wasp.sift.detection import Detection


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b.T)  # a [N,D], b [M,D]


def non_maximum_averaging(
    detections: list[Detection],
    iou_threshold: float = 0.01,
) -> list[Detection]:
    if not detections:
        return []

    boxes = np.array([det.coords for det in detections], dtype=np.float32)
    sims = np.array([det.similarity for det in detections])
    classes = [det.class_name for det in detections]

    # Compute areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    merged = []
    used = np.zeros(len(detections), dtype=bool)

    for i in range(len(detections)):
        if used[i]:
            continue

        # Start a new cluster
        cluster_idxs = [i]
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue

            # Compute IoU between box i and j
            x1 = max(boxes[i, 0], boxes[j, 0])
            y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2])
            y2 = min(boxes[i, 3], boxes[j, 3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = areas[i] + areas[j] - inter
            iou = inter / union if union > 0 else 0.0

            if iou >= iou_threshold and classes[i] == classes[j]:
                cluster_idxs.append(j)
                used[j] = True

        # Average coordinates and similarity within cluster
        cluster_boxes = boxes[cluster_idxs]
        cluster_sims = sims[cluster_idxs]
        avg_box = cluster_boxes.mean(axis=0)
        avg_sim = cluster_sims.mean()

        merged.append(
            Detection(
                class_name=classes[i],
                similarity=float(avg_sim),
                coords=tuple(map(int, avg_box)),  # type: ignore
            )
        )

        used[i] = True

    return merged
