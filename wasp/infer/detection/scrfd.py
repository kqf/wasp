import cv2
import numpy as np
import onnxruntime

from wasp.infer.distance import distance2bbox, distance2kps


def nms(dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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


class SCRFD:
    def __init__(
        self,
        model_file=None,
        nms_thresh=0.4,
        det_thresh=0.5,
        input_size=(640, 640),
        batched=False,
    ):
        self.model_file = model_file
        self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.batched = batched
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        self.det_thresh = det_thresh
        self._init_vars()
        self.input_size = input_size

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        # print(input_shape)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        # print('image_size:', self.image_size)
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = [o.name for o in outputs]
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        # print(self.output_names)
        # assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs) == 6 or len(outputs) != 9 and len(outputs) == 10:
            self._extracted_from__init_vars_()
        elif len(outputs) in {9, 15}:
            self._extracted_from__init_vars_()
            self.use_kps = True

    # TODO Rename this here and in `_init_vars`
    def _extracted_from__init_vars_(self):  # noqa <<<<<<<<<<<<
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get("det_thresh", None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("Det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def forward(self, image, threshold):  # sourcery skip: low-code-quality
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[:2][::-1])
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        image: np.ndarray,
        input_size=None,
        max_num=0,
        metric="default",
        det_thresh=0.5,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        input_size = input_size or self.input_size
        det_img, det_scale = resize_image(image, input_size)
        pre_det, kpss = detect_objects(
            self.forward,
            det_img,
            det_thresh,
            det_scale,
            self.nms_thresh,
        )
        det, kpss = filter_objects(
            pre_det,
            kpss,
            max_num,
            image.shape,
            metric,
        )
        return det, kpss


def resize_image(img, input_size):
    im_ratio = img.shape[0] / img.shape[1]
    model_ratio = input_size[1] / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = new_height / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    return det_img, det_scale


def detect_objects(
    forward,
    det_img,
    det_thresh,
    det_scale,
    nms_thresh,
):
    scores_list, bboxes_list, kpss_list = forward(det_img, det_thresh)
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det, threshold=nms_thresh)
    pre_det = pre_det[keep, :]
    kpss = kpss[order, :, :]
    kpss = kpss[keep, :, :]
    return pre_det, kpss


def filter_objects(
    pre_det,
    kpss,
    max_num,
    img_shape,
    metric="default",
):
    if max_num <= 0 or pre_det.shape[0] <= max_num:
        return pre_det, kpss

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
    if kpss is not None:
        kpss = kpss[bindex, :]

    return pre_det, kpss
