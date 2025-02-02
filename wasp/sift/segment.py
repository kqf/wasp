from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms


def load_image(image_path):
    return cv2.imread(str(image_path))


def load_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model


def infer(image, model):
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    return output.argmax(0).byte().cpu().numpy()


def expand_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def save_mask(output_predictions, output_path):
    np.save(output_path, output_predictions)


def display_images(original_image, expanded_mask):
    num_classes = expanded_mask.max() + 1
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

    colored_mask = colors[expanded_mask]
    colored_mask_cv = cv2.resize(
        colored_mask, (original_image.shape[1], original_image.shape[0])
    )
    blended = cv2.addWeighted(original_image, 0.6, colored_mask_cv, 0.4, 0)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Segmented Image", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model = load_model()
    path = Path("datasets/a")
    for file in path.glob("*.jpg"):
        image = load_image(file)
        predictions = infer(image, model)
        expanded_predictions = expand_mask(predictions, kernel_size=10)
        save_mask(expanded_predictions.clip(0, 1), file.with_suffix(".npy"))
        display_images(image, expanded_predictions)


if __name__ == "__main__":
    main()
