import numpy as np


def random_crop(*args, **kwargs):
    return *args


class Preproc:
    def __init__(self, img_dim: int) -> None:
        self.img_dim = img_dim

    def __call__(
        self,
        image: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if targets.shape[0] == 0:
            raise ValueError("this image does not have gt")

        boxes = targets[:, :4].copy()
        landmarks = targets[:, 4:-1].copy()
        labels = targets[:, -1:].copy()

        image_t, boxes_t, labels_t, landmarks_t, pad_image_flag = random_crop(
            image, boxes, labels, landmarks, self.img_dim
        )

        image_t = _pad_to_square(image_t, pad_image_flag)
        image_t, boxes_t, landmarks_t = random_horizontal_flip(
            image_t, boxes_t, landmarks_t
        )
        height, width = image_t.shape[:2]

        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height

        landmarks_t[:, 0::2] = landmarks_t[:, 0::2] / width
        landmarks_t[:, 1::2] = landmarks_t[:, 1::2] / height

        targets_t = np.hstack((boxes_t, landmarks_t, labels_t))

        return image_t, targets_t
