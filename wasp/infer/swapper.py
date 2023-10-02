import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

from wasp.infer.detection.nn import nninput
from wasp.infer.distance import norm_crop


class INSwapper:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.session = session or onnxruntime.InferenceSession(
            self.model_file,
            None,
        )
        inputs = self.session.get_inputs()
        self.input_names = []
        self.input_names.extend(inp.name for inp in inputs)
        outputs = self.session.get_outputs()
        output_names = [out.name for out in outputs]
        self.output_names = output_names
        assert len(self.output_names) == 1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print("inswapper-shape:", self.input_shape, output_shape)
        self.input_size = tuple(input_shape[2:4][::-1])
        print(self.input_size)

    def get(self, img, target_face, source_face, paste_back=False):
        aimg, M = norm_crop(img, target_face.kps, self.input_size[0])
        blob = nninput(
            aimg,
            std=255.0,
            mean=0.0,
            shape=self.input_size,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(
            self.output_names,
            {
                self.input_names[0]: blob.astype(np.float32),
                self.input_names[1]: latent.astype(np.float32),
            },
        )[0]
        # print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return self.prepare_features(img, bgr_fake, aimg, M)

    # TODO Rename this here and in `get`
    def prepare_features(self, img, bgr_fake, aimg, M):
        target_img = img
        fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0
        IM = cv2.invertAffineTransform(M)
        img_white = np.full(
            (aimg.shape[0], aimg.shape[1]),
            255,
            dtype=np.float32,
        )
        bgr_fake = cv2.warpAffine(
            bgr_fake,
            IM,
            (target_img.shape[1], target_img.shape[0]),
            borderValue=0.0,
        )
        img_white = cv2.warpAffine(
            img_white,
            IM,
            (target_img.shape[1], target_img.shape[0]),
            borderValue=0.0,
        )
        fake_diff = cv2.warpAffine(
            fake_diff,
            IM,
            (target_img.shape[1], target_img.shape[0]),
            borderValue=0.0,
        )
        img_white[img_white > 20] = 255
        fthresh = 10
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        # mask_h = np.max(mask_h_inds, 10) - np.min(mask_h_inds, 10)
        # mask_w = np.max(mask_w_inds, 10) - np.min(mask_w_inds, 10)
        # mask_size = int(np.sqrt(mask_h * mask_w))
        mask_size = 100
        k = max(mask_size // 10, 10)
        # k = max(mask_size//20, 6)
        # k = 6
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel = np.ones((2, 2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
        k = max(mask_size // 20, 5)
        # k = 3
        # k = 3
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        img_mask /= 255
        fake_diff /= 255
        # img_mask = fake_diff
        img_mask = np.reshape(
            img_mask,
            [img_mask.shape[0], img_mask.shape[1], 1],
        )
        fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(
            np.float32
        )
        fake_merged = fake_merged.astype(np.uint8)
        return fake_merged
