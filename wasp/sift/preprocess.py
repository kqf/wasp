import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Strong bilateral filter — preserves edges, smooths texture
    filtered = cv2.bilateralFilter(lab, d=15, sigmaColor=80, sigmaSpace=80)

    # Convert back to RGB
    rgb = cv2.cvtColor(filtered, cv2.COLOR_LAB2RGB)

    # Optional: downscale + upscale — further remove tiny details
    h, w = rgb.shape[:2]
    rgb_small = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(rgb_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Optional slight morphological closing — fill small holes
    kernel = np.ones((3, 3), np.uint8)
    rgb = cv2.morphologyEx(rgb, cv2.MORPH_CLOSE, kernel)

    return rgb


def main():
    image = cv2.imread("datasets/test-new/1.png")
    preprocessed = preprocess(image)
    cv2.imshow("image", preprocessed)
    cv2.waitKey()


if __name__ == "__main__":
    main()
