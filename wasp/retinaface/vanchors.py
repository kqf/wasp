import matplotlib.pyplot as plt
import torch
from torchvision.models.detection.image_list import ImageList

from wasp.retinaface.ssd import ssdlite320_mobilenet_v3_large_custom


def visualize_anchors(anchors, image_size):
    """Visualize anchor boxes on an image grid."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_xlim(0, image_size[1])
    ax.set_ylim(0, image_size[0])
    ax.invert_yaxis()

    for anchor in anchors:
        x, y, w, h = anchor.tolist()
        rect = plt.Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
        ax.add_patch(rect)

    plt.title("Anchor Boxes")
    # plt.show()
    plt.savefig("anchors.png")


def main():
    # Define model input size
    input_size = 640, 640

    # Initialize the model using your custom function
    model = ssdlite320_mobilenet_v3_large_custom(
        size=input_size, num_classes=2
    )  # Change num_classes as needed

    input_tensor = torch.randn(
        1, 3, input_size[0], input_size[1]
    )  # Change if different input size

    # Extract features using the model's backbone
    backbone = model.backbone
    features = backbone(input_tensor)

    # Create an ImageList object for the anchor generator
    image_list = ImageList(input_tensor, [input_size])

    # Generate anchor boxes using the anchor generator
    feature_maps = list(features.values())
    anchors = model.anchor_generator(image_list, feature_maps)

    # Flatten anchors to visualize
    flattened_anchors = [anchor.reshape(-1, 4) for anchor in anchors]
    flattened_anchors = torch.cat(
        flattened_anchors, dim=0
    )  # Concatenate all anchors for visualization
    visualize_anchors(flattened_anchors, input_size)


if __name__ == "__main__":
    main()
