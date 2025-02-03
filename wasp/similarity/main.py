import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

from wasp.retinaface.priors import priorbox


class ResNet50Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.backbone(x)


def plot(image: np.ndarray, priors: torch.Tensor):
    h, w = image.shape[:2]

    for box in priors:
        cx, cy, bw, bh = box.tolist()
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    cv2.imshow("Image with Prior Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_blob(image: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)


def main():
    model = ResNet50Backbone()

    resolutions = [(224, 224), (640, 480), (480, 640)]

    for resolution in resolutions:
        image = np.random.randint(0, 256, (*resolution, 3), dtype=np.uint8)

        features = model(to_blob(image))
        print(f"Feature map shape for {resolution}: {features.shape}")

        priors = priorbox(
            min_sizes=[[resolution[0] // 8]],
            steps=[8],
            clip=True,
            image_size=resolution,
        )
        print(priors.shape)

        # Plot using OpenCV
        plot(image, priors)


if __name__ == "__main__":
    main()
