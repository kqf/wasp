from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T


@dataclass
class MaskRCNNOutput:
    masks: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    masks_dilated: torch.Tensor = None


def load_model():
    weights = (
        torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def load_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0), image


def infer(image_tensor, model) -> MaskRCNNOutput:
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return MaskRCNNOutput(
        masks=predictions["masks"],
        labels=predictions["labels"],
        scores=predictions["scores"],
        boxes=predictions["boxes"],
    )


def expand_mask(predictions: MaskRCNNOutput, kernel_size=10) -> MaskRCNNOutput:
    masks = predictions.masks.squeeze(1).detach().cpu().numpy()
    masks = (masks > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_masks = [cv2.dilate(mask, kernel, iterations=1) for mask in masks]
    return MaskRCNNOutput(
        masks=predictions.masks,
        labels=predictions.labels,
        scores=predictions.scores,
        boxes=predictions.boxes,
        masks_dilated=torch.Tensor(expanded_masks).unsqueeze(1),
    )


def save_mask(mask, output_path):
    np.save(output_path, mask.cpu().numpy())


def display_images(image, predictions: MaskRCNNOutput):
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    for m in predictions.masks_dilated.squeeze(1).detach().cpu().numpy():
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        for i in range(3):
            mask_overlay[:, :, i] = np.where(
                m > 0, color[i], mask_overlay[:, :, i]
            )
    blended = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)
    for i, box in enumerate(predictions.boxes.detach().cpu().numpy()):
        if predictions.scores[i] < 0.8:
            continue
        print(predictions.scores[i])
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow(
        "Mask R-CNN Inference", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model = load_model()
    path = Path("datasets/a")
    for file in path.glob("*.jpg"):
        image_tensor, image = load_image(file)
        predictions = infer(image_tensor, model)
        expanded_predictions = expand_mask(predictions, kernel_size=10)
        save_mask(expanded_predictions.masks_dilated, file.with_suffix(".npy"))
        display_images(image, expanded_predictions)


if __name__ == "__main__":
    main()
