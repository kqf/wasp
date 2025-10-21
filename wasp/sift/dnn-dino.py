from pathlib import Path

import cv2
import torch
from groundingdino.util.inference import load_model, predict  # type: ignore
from groundingdino.util.vl_utils import annotate  # type: ignore


def main():
    # Choose the model and text prompt
    model_name = "GroundingDINO_SwinT_OGC"
    text_prompt = "a bird, an eagle, cartoon bird"

    # Use Metal (MPS) for Apple Silicon, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model = load_model(model_name, device=device)

    # Set thresholds
    box_threshold = 0.35
    text_threshold = 0.25

    for file in Path("datasets/test-new").glob("*.png"):
        print(f"Processing {file.name}")
        img_bgr = cv2.imread(str(file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run prediction
        boxes, logits, phrases = predict(
            model=model,
            image_source=img_rgb,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Annotate the detections
        annotated_frame = annotate(
            image_source=img_rgb.copy(),
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )

        # Convert back to BGR for OpenCV display/saving
        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        outpath = f"{file.stem}-groundingdino.png"
        cv2.imwrite(outpath, annotated_bgr)
        print(f"Saved {outpath}")

        # Optional display
        cv2.imshow("Grounding DINO", annotated_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
