import functools

import cv2
import numpy as np
import onnxruntime

from wasp.infer.distance import distance2bbox, distance2kps


def nms(scores, dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


@functools.lru_cache
def anchors_centers(
    height: int,
    width: int,
    stride: int,
    num_anchors: int,
) -> np.ndarray:
    anchor_centers = np.stack(
        np.mgrid[:height, :width][::-1],
        axis=-1,
    ).astype(np.float32)
    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
    if num_anchors <= 1:
        return anchor_centers

    anchor_centers = np.stack(
        [anchor_centers] * num_anchors,
        axis=1,
    ).reshape((-1, 2))
    return anchor_centers


def nninput(
    image,
    mean: float = 127.5,
    std: float = 128.0,
) -> np.ndarray:
    *shape, _ = image.shape
    return cv2.dnn.blobFromImage(
        image,
        1.0 / std,
        shape,
        (mean, mean, mean),
        swapRB=True,
    )


class SCRFD:
    def __init__(
        self,
        model_file=None,
        nms_thresh=0.4,
        det_thresh=0.5,
        input_size=(640, 640),
    ):
        self.model_file = model_file
        self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        self.det_thresh = det_thresh
        self.input_size = input_size
        input_cfg = self.session.get_inputs()[0]
        outputs = self.session.get_outputs()
        output_names = [o.name for o in outputs]
        self.input_name = input_cfg.name
        self.output_names = output_names
        self._feat_stride_fpn = [8, 16, 32]
        self._anchor_ratio = 1.0
        self._num_anchors = 2

    def forward(
        self,
        image: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # blob is of shape [b=1, c, h, w]
        blob = nninput(image)

        net_outs: list[np.ndarray] = self.session.run(
            self.output_names, {self.input_name: blob}
        )
        n = len(net_outs) // len(self._feat_stride_fpn)
        scores, boxes, keypoints = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            score = net_outs[idx]
            bbox = net_outs[idx + n] * stride
            keypoint = net_outs[idx + n * 2] * stride
            anchors = anchors_centers(
                blob.shape[2] // stride,
                blob.shape[3] // stride,
                stride,
                self._num_anchors,
            )

            pos_inds = np.where(score >= threshold)[0]
            bboxes = distance2bbox(anchors, bbox)
            pos_scores = score[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores.append(pos_scores)
            boxes.append(pos_bboxes)

            kpss = distance2kps(anchors, keypoint)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            keypoints.append(pos_kpss)

        return np.vstack(scores), np.vstack(boxes), np.vstack(keypoints)

    def detect(
        self,
        image: np.ndarray,
        input_size=None,
        max_num=0,
        metric="default",
        det_thresh=0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_size = input_size or self.input_size
        det_img, backscale = resize_image(image, input_size)
        scores, boxes, keypts = self.forward(det_img, det_thresh)
        scores, pre_det, kpss = filter_objects(
            scores,
            boxes / backscale,
            keypts / backscale,
            self.nms_thresh,
        )
        scores, det, kpss = remove_small_objects(
            scores,
            pre_det,
            kpss,
            max_num,
            image.shape,
            metric,
        )
        return scores, det, kpss


def resize_image(
    image: np.ndarray,
    isize: tuple[int, int],
) -> tuple[np.ndarray, float]:
    img_height, img_width, _ = image.shape
    target_width, target_height = isize

    # Calculate the scaling factors
    width_scale = target_width / img_width
    height_scale = target_height / img_height

    # Choose the scaling factor that preserves the aspect ratio
    scale = min(width_scale, height_scale)

    # Calculate the new dimensions
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    resized_img = cv2.resize(image, (new_width, new_height))

    # Create a black canvas of the target size
    det_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # Copy the resized image to the canvas
    det_img[:new_height, :new_width, :] = resized_img
    return det_img, scale


def filter_objects(
    scores: np.ndarray,
    bboxes: np.ndarray,
    keypts: np.ndarray,
    nms_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = scores.ravel().argsort()[::-1]

    scores = scores[order]
    bboxes = bboxes[order, :]
    keypts = keypts[order, :, :]
    keep = nms(scores, bboxes, threshold=nms_thresh)

    scores = scores[keep]
    bboxes = bboxes[keep, :]
    keypts = keypts[keep, :, :]
    return scores, bboxes, keypts


def remove_small_objects(
    scores,
    pre_det,
    kpss,
    max_num,
    img_shape,
    metric="default",
):
    if max_num <= 0 or pre_det.shape[0] <= max_num:
        return scores, pre_det, kpss

    area = (pre_det[:, 2] - pre_det[:, 0]) * (pre_det[:, 3] - pre_det[:, 1])
    img_center = (img_shape[0] // 2, img_shape[1] // 2)
    offsets = np.vstack(
        [
            (pre_det[:, 0] + pre_det[:, 2]) / 2 - img_center[1],
            (pre_det[:, 1] + pre_det[:, 3]) / 2 - img_center[0],
        ]
    )
    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0) * 2.0
    values = area if metric == "max" else (area - offset_dist_squared)
    bindex = np.argsort(values)[::-1]
    bindex = bindex[:max_num]
    pre_det = pre_det[bindex, :]
    kpss = kpss[bindex, :]
    scores = scores[bindex]

    return scores, pre_det, kpss
