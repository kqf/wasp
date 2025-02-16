import cv2
import numpy as np
import torch
from face_alignment import align

from models import get_face_alignment_net


def load_model(model_name="mobilenet_1.0_192x192"):
    model = get_face_alignment_net(model_name)
    model.load_state_dict(
        torch.load(f"pretrained/{model_name}.pth", map_location="cpu")
    )
    model.eval()
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (192, 192))
    image_tensor = (
        torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
        / 255.0
    )
    return image, image_tensor


def predict_landmarks(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    return output.squeeze().numpy().reshape(-1, 2) * 192


def draw_landmarks(image, landmarks):
    for x, y in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite("output_landmarks.jpg", image)


def main():
    model = load_model("mobilenet_1.0_192x192")
    image, image_tensor = preprocess_image("tests/assets/lenna.png")
    landmarks = predict_landmarks(model, image_tensor)
    draw_landmarks(image, landmarks)


if __name__ == "__main__":
    main()
