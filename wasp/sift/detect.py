from dataclasses import dataclass
from pathlib import Path

import cv2
import timm
import torch
import torchvision.transforms as T


@dataclass
class EVA02Output:
    labels: torch.Tensor
    scores: torch.Tensor


def load_model():
    """Load the EVA-02 model."""
    model = timm.create_model("eva02_large_patch14_224", pretrained=True)
    model.eval()
    return model


def load_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),  # Resize for EVA-02
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),  # Standard normalization
        ]
    )
    return transform(image).unsqueeze(0), image


def infer(image_tensor, model) -> EVA02Output:
    """Run inference using EVA-02 model."""
    with torch.no_grad():
        logits = model(image_tensor)
        scores = torch.nn.functional.softmax(logits, dim=1)
        top_scores, top_labels = torch.topk(
            scores, k=5
        )  # Get top 5 predictions
    return EVA02Output(
        labels=top_labels.squeeze(0), scores=top_scores.squeeze(0)
    )


def display_results(image, predictions: EVA02Output):
    """Display image with predicted labels."""
    for i, (label, score) in enumerate(
        zip(predictions.labels, predictions.scores)
    ):
        text = f"Class {label.item()} - {score.item():.2f}"
        cv2.putText(
            image,
            text,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("EVA-02 Inference", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model = load_model()
    path = Path("datasets/a")

    for file in path.glob("*.jpg"):
        image_tensor, image = load_image(file)
        predictions = infer(image_tensor, model)
        display_results(image, predictions)


if __name__ == "__main__":
    main()
