from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T


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


def infer(image_tensor, model):
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return predictions


def expand_mask(predictions, kernel_size=10):
    masks = predictions["masks"].squeeze(1).detach().cpu().numpy()
    masks = (masks > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_masks = [cv2.dilate(mask, kernel, iterations=1) for mask in masks]
    predictions["masks_dilated"] = torch.Tensor(expanded_masks).unsqueeze(0)
    return predictions


def save_mask(mask, output_path):
    np.save(output_path, mask)


def display_images(image, predictions):
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    for m in predictions["masks"]:
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        for i in range(3):
            mask_overlay[:, :, i] = np.where(
                m > 0, color[i], mask_overlay[:, :, i]
            )
    blended = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)
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
        save_mask(
            expanded_predictions["masks_dilated"].clip(0, 1),
            file.with_suffix(".npy"),
        )
        display_images(image, expanded_predictions)


if __name__ == "__main__":
    main()
