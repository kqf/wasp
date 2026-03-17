import cv2
import torch
from torchvision.models.detection import retinanet_mobilenet_v3_large_fpn_v2
from torchvision.transforms import functional as F


def main():
    image = cv2.imread("image.jpg")
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    crop = image[cy - 60 : cy + 60, cx - 60 : cx + 60]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(crop)
    model = retinanet_mobilenet_v3_large_fpn_v2(weights="DEFAULT")
    model.eval()
    with torch.no_grad():
        outputs = model([tensor])
    print(outputs)


if __name__ == "__main__":
    main()
