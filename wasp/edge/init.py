import cv2
import torch
from torchvision.models.detection import retinanet_mobilenet_v3_large_fpn_v2
from torchvision.transforms import functional as F


def center_crop(image, size=120):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = size // 2
    return image[cy - half : cy + half, cx - half : cx + half]


def to_blob(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return F.to_tensor(image)


def main():
    image = cv2.imread("image.jpg")
    crop = center_crop(image, 120)
    tensor = to_blob(crop)
    model = retinanet_mobilenet_v3_large_fpn_v2(weights="DEFAULT")
    model.eval()
    with torch.no_grad():
        outputs = model([tensor])
    print(outputs)


if __name__ == "__main__":
    main()
