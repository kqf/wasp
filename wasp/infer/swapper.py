from functools import partial

import cv2
import nptyping as npt
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

from wasp.face import Face
from wasp.infer.distance import norm_crop
from wasp.infer.nn import nninput


def _diff(bgr_fake, aimg) -> np.ndarray:
    fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0
    return fake_diff


def warp(image: np.ndarray, IM: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.warpAffine(
        image,
        IM,
        (shape[1], shape[0]),
        borderValue=0.0,
    )


class INSwapper:
    def __init__(
        self,
        model_file=None,
        session=None,
        resolution: tuple[int, int] = (
            128,
            128,
        ),
    ):
        self.resolution = resolution
        self.emap = numpy_helper.to_array(
            onnx.load(model_file).graph.initializer[-1],
        )
        self.session = session or onnxruntime.InferenceSession(
            model_file,
            None,
        )
        inputs = self.session.get_inputs()
        self.input_names: list[str] = []
        self.input_names.extend(inp.name for inp in inputs)
        self.output_names = [out.name for out in self.session.get_outputs()]

    def get(
        self,
        image: np.ndarray,
        target: Face,
        source: Face,
    ) -> np.ndarray:
        # Crop a single face:
        crop, M = norm_crop(image, target.kps, self.resolution[0])
        blob: npt.NDArray[npt.Shape["1, 3, 128, 128"]] = nninput(
            crop,
            std=255.0,
            mean=0.0,
            shape=self.resolution,
        )
        # latent[1, 512]
        latent = source.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        print(latent.shape)

        pred: npt.NDArray[npt.Shape["1, 3, 128, 128"]] = self.session.run(
            self.output_names,
            {
                self.input_names[0]: blob.astype(np.float32),
                self.input_names[1]: latent.astype(np.float32),
            },
        )[0]
        # Switch to channels last
        ch_laset = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * ch_laset, 0, 255).astype(np.uint8)[:, :, ::-1]
        return self.blend(image.copy(), bgr_fake, crop, M)

    def blend(self, image, bgr_fake, crop, M):
        IM = cv2.invertAffineTransform(M)
        white = np.full((crop.shape[0], crop.shape[1]), 255, dtype=np.float32)
        warps = partial(warp, IM=IM, shape=image.shape)
        bgr_f = warps(bgr_fake)
        white = warps(white)
        fake_diff = warps(_diff(bgr_fake, crop))
        white[white > 20] = 255
        fthresh = 10
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        img_mask = white
        # mask_size = int(np.sqrt(mask_h * mask_w))
        mask_size = 100
        k = max(mask_size // 10, 10)
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
        fake_merged = img_mask * bgr_f + (1 - img_mask) * image.astype(
            np.float32,
        )
        fake_merged = fake_merged.astype(np.uint8)
        return fake_merged
