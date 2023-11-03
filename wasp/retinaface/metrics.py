import numpy as np


def group_by_key(gt, image):
    return gt


def get_overlaps(gt_boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    i_xmin = np.maximum(gt_boxes[:, 0], box[0])
    i_ymin = np.maximum(gt_boxes[:, 1], box[1])

    gt_xmax = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_ymax = gt_boxes[:, 1] + gt_boxes[:, 3]

    box_xmax = box[0] + box[2]
    box_ymax = box[1] + box[3]

    i_xmax = np.minimum(gt_xmax, box_xmax)
    i_ymax = np.minimum(gt_ymax, box_ymax)

    iw = np.maximum(i_xmax - i_xmin, 0.0)
    ih = np.maximum(i_ymax - i_ymin, 0.0)

    intersection = iw * ih

    union = box[2] * box[3] + gt_boxes[:, 2] * gt_boxes[:, 3] - intersection

    return intersection / (union + 1e-7)


def get_envelope(precisions: np.ndarray) -> np.array:
    """Compute the envelope of the precision curve.

    Args:
      precisions:

    Returns: enveloped precision

    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate area under precision/recall curve.

    Args:
      recalls:
      precisions:

    Returns:

    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve,
    # look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    return np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])


def recall_precision(
    gt: np.ndarray,
    predictions: np.ndarray,
    iou_threshold: float,
) -> tuple[np.array, np.array, np.array]:
    num_gts = len(gt)
    image_gts = group_by_key(gt, "image_id")

    image_gt_boxes = {
        img_id: np.array([[float(z) for z in b["bbox"]] for b in boxes])
        for img_id, boxes in image_gts.items()
    }
    image_gt_checked = {
        img_id: np.zeros(len(boxes)) for img_id, boxes in image_gts.items()
    }

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)

    for prediction_index, prediction in enumerate(predictions):
        box = prediction["bbox"]

        max_overlap = -np.inf
        jmax = -1

        try:
            # gt_boxes per image
            gt_boxes = image_gt_boxes[prediction["image_id"]]

            # gt flags per image
            gt_checked = image_gt_checked[prediction["image_id"]]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if gt_boxes:
            overlaps = get_overlaps(gt_boxes, box)

            max_overlap = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if max_overlap >= iou_threshold and gt_checked[jmax] == 0:
            tp[prediction_index] = 1.0
            gt_checked[jmax] = 1
        else:
            fp[prediction_index] = 1.0
    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)

    # avoid zero div in case the first detection matches
    # a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = get_ap(recalls, precisions)
    return recalls, precisions, ap
