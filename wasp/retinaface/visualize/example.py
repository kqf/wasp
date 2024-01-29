import albumentations as alb
import cv2
import matplotlib.pyplot as plt
import numpy as np

from wasp.retinaface.data import Annotation
from wasp.retinaface.visualize.plot import plot


def main():
    image = cv2.cvtColor(
        cv2.imread("couple.jpg"),
        cv2.COLOR_BGR2RGB,
    )

    boxes = (
        (332, 128, 542, 424),
        (542, 232, 726, 498),
    )
    keypoints = (
        [
            [410.562, 223.625],
            [482.817, 268.089],
            [436.5, 286.616],
            [364.246, 301.438],
            [443.911, 344.049],
        ],
        [
            [590.205, 329.531],
            [676.795, 337.857],
            [633.5, 381.152],
            [580.214, 417.786],
            [668.469, 429.442],
        ],
    )

    transform = alb.Compose(
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
        ),
        keypoint_params=alb.KeypointParams(
            format="xy",
        ),
        p=1,
        transforms=[
            # alb.Resize(height=1024, width=1024, p=1),
        ],
    )

    sample = transform(
        image=image,
        bboxes=boxes,
        category_ids=np.ones(len(boxes)),
        keypoints=np.asarray(keypoints).reshape(-1, 2),
    )

    image = sample["image"]
    boxes = sample["bboxes"]
    transofrmed_keypoints = np.asarray(sample["keypoints"])
    keypoints = transofrmed_keypoints.reshape(-1, 5, 2).tolist()
    transformed = [Annotation(b, k) for b, k in zip(boxes, keypoints)]

    plt.imshow(plot(image, annotations=transformed))
    plt.show()


if __name__ == "__main__":
    main()
