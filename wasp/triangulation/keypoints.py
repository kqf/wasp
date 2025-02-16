import cv2
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework


def main():
    model_config = ModelConfig("wflw")
    model_config.model_weights_path = "."
    model = SPIGAFramework(model_config, gpus=["cpu"])

    image_path = "tests/assets/lenna.png"
    image = cv2.imread(image_path)
    print(image.shape)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bbox = [0, 0, image_rgb.shape[1], image_rgb.shape[0]]
    results = model.inference(image_rgb, [bbox])

    landmarks = results["landmarks"][0]

    for x, y in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow("output", image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
