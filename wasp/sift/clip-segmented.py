import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import clip
import cv2
import numpy as np
import torch
import tqdm
from dataclasses_json import dataclass_json
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


@dataclass
class Proposal:
    roi: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    mask: np.ndarray  # boolean mask of the object


def segment(image: np.ndarray) -> List[Proposal]:
    # Load SAM model
    sam_checkpoint = (
        # "sam_vit_h_4b8939.pth"  # download from SAM repo if you don't have it
        # "sam_vit_l_0b3195.pth"
        "sam_vit_b_01ec64.pth"
    )
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=64,
    )

    # Generate masks
    masks = mask_generator.generate(image)

    proposals = []
    for m in masks:
        mask = m["segmentation"]  # boolean mask
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        roi = (xmin, ymin, xmax, ymax)

        proposals.append(Proposal(roi=roi, mask=mask))

    return proposals


def dump_database(output: str, descriptors) -> None:
    np.savez_compressed(output, descriptors=descriptors)


def build_text_features(
    class_prompts: dict[str, str],
    model_name="ViT-B/32",
) -> dict[str, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)

    text_features = {}
    for class_name, prompt in class_prompts.items():
        with torch.no_grad():
            tokens = clip.tokenize([prompt]).to(device)
            text_emb = model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_features[class_name] = (
                text_emb.cpu().numpy().astype(np.float32)
            )
        print(f"Built text embedding for '{class_name}' -> {prompt}")

    return text_features


@dataclass_json
@dataclass
class Detection:
    class_name: str
    similarity: float
    coords: tuple[int, int, int, int]


def save_detections(
    detections: list[Detection], output_path: Path | str = "test.json"
):
    with open(output_path, "w") as f:
        f.write(
            Detection.schema().dumps(detections, many=True),  # type: ignore
        )


def load_detections(input_path: Path | str = "test.json") -> list[Detection]:
    with open(input_path, "r") as f:
        data = json.load(f)
    return [Detection.from_dict(d) for d in data]  # type: ignore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b.T)  # a [N,D], b [M,D]


def masked_crop_from_proposal(
    image: np.ndarray, proposal: Proposal
) -> np.ndarray:
    xmin, ymin, xmax, ymax = proposal.roi
    # Safety checks
    h, w = image.shape[:2]
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    # Crop the image and corresponding mask
    crop = image[ymin:ymax, xmin:xmax].copy()
    mask_crop = proposal.mask[ymin:ymax, xmin:xmax]

    # Expand mask to 3 channels for RGB masking
    mask_crop_3c = np.repeat(mask_crop[:, :, None], 3, axis=2)

    # Apply mask (zero background)
    crop_masked = crop * mask_crop_3c

    return crop_masked.astype(np.uint8)


def detect_objects(
    input_image: np.ndarray,
    stacked_databases: dict[str, np.ndarray],
    model_name: str = "ViT-B/32",
    similarity_threshold: float = 0.25,
    show_patches: bool = False,
) -> list[Detection]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    proposals = segment(rgb)
    detections: list[Detection] = []

    # Generate all patch crops
    all_patches, all_coords = [], []
    for proposal in tqdm.tqdm(proposals):
        crop = masked_crop_from_proposal(rgb, proposal)
        if show_patches:
            cv2.imshow("patch", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cv2.waitKey()

        x = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        all_patches.append(emb.cpu().numpy())
        all_coords.append(proposal)

    if not all_patches:
        return []

    vectors = np.vstack(all_patches).astype("float32")

    # Compare each patch embedding with each class database
    for class_name, class_db in stacked_databases.items():
        sims = cosine_similarity(vectors, class_db)
        max_sim = sims.max(axis=1)
        print(max_sim)

        detections.extend(
            Detection(
                class_name=class_name,
                similarity=float(sim),
                coords=all_coords[idx].roi,
            )
            for idx, sim in enumerate(max_sim)
            if sim >= similarity_threshold
        )
    return detections


def visualize_detections(
    input_image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    output_image = input_image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.coords
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output_image,
            f"{det.class_name}:{det.similarity:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output_image


def non_maximum_averaging(
    detections: list[Detection],
    iou_threshold: float = 0.01,
) -> list[Detection]:
    if not detections:
        return []

    boxes = np.array([det.coords for det in detections], dtype=np.float32)
    sims = np.array([det.similarity for det in detections])
    classes = [det.class_name for det in detections]

    # Compute areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    merged = []
    used = np.zeros(len(detections), dtype=bool)

    for i in range(len(detections)):
        if used[i]:
            continue

        # Start a new cluster
        cluster_idxs = [i]
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue

            # Compute IoU between box i and j
            x1 = max(boxes[i, 0], boxes[j, 0])
            y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2])
            y2 = min(boxes[i, 3], boxes[j, 3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = areas[i] + areas[j] - inter
            iou = inter / union if union > 0 else 0.0

            if iou >= iou_threshold and classes[i] == classes[j]:
                cluster_idxs.append(j)
                used[j] = True

        # Average coordinates and similarity within cluster
        cluster_boxes = boxes[cluster_idxs]
        cluster_sims = sims[cluster_idxs]
        avg_box = cluster_boxes.mean(axis=0)
        avg_sim = cluster_sims.mean()

        merged.append(
            Detection(
                class_name=classes[i],
                similarity=float(avg_sim),
                coords=tuple(map(int, avg_box)),  # type: ignore
            )
        )

        used[i] = True

    return merged


# This needs to be installed:
# pip install git+https://github.com/facebookresearch/segment-anything.git
# Then see the github to download the models


def plot_and_save(name, timestamp, image):
    cv2.imwrite(f"{name}{timestamp}.png", image)
    cv2.imshow("output", image)
    cv2.waitKey()


def main():
    # Instead of image-based databases, we now use text prompts:
    class_prompts = {
        "R": "a photo of the letter R",
        "A": "a picture of the letter A",
    }

    stacked_databases = build_text_features(class_prompts)

    # Save them (optional)
    for name, base in stacked_databases.items():
        dump_database(f"database-{name}.npz", base)

    # Prepare output folder
    outpath = Path("datasets/test/v2-CLIP-TEXT")
    outpath.mkdir(parents=True, exist_ok=True)

    # Run inference for each test image
    for file in Path("datasets/test-new").glob("*.png"):
        image = cv2.imread(str(file))
        detections = detect_objects(image, stacked_databases)
        print(detections)

        annotated_raw = visualize_detections(image, detections)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_and_save("raw-", timestamp, annotated_raw)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
