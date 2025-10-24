from datetime import datetime
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
from PIL import Image

from wasp.sift.detection import Detection, load_detections, save_detections


def dump_database(output: str, descriptors) -> None:
    np.savez_compressed(output, descriptors=descriptors)


def build_features(
    impath: str,
    model_name="ViT-B/32",
    average=True,
) -> np.ndarray:
    path = Path(impath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    embs = []
    for file in path.glob("*.png"):
        img = Image.open(file).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb.cpu().numpy())

    if not embs:
        raise RuntimeError(f"No .png files found in {impath}")

    # Compute the mean embedding along the 0-th axis
    vectors = np.vstack(embs).astype(np.float32)
    print(f"Database shape: {vectors.shape=}")
    if average:
        return np.mean(vectors, axis=0, keepdims=True).astype(np.float32)
    return vectors


def sliding_patches(
    resolution: tuple[int, int],
    patch_sizes=(96, 128, 160, 224),  # added smaller scales
    stride_ratio=0.4,  # denser stride for small patches
):
    W, H = resolution
    for ps in patch_sizes:
        stride = int(ps * stride_ratio)
        for x in range(0, W - ps + 1, stride):
            for y in range(0, H - ps + 1, stride):
                yield (x, y, x + ps, y + ps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b.T)  # a [N,D], b [M,D]


def detect_objects(
    input_image: np.ndarray,
    stacked_databases: dict[str, np.ndarray],
    model_name: str = "ViT-B/32",
    similarity_threshold: float = 0.90,
    show_patches: bool = False,
) -> list[Detection]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    detections: list[Detection] = []

    # Generate all patch crops
    all_patches, all_coords = [], []
    for coords in sliding_patches(pil_img.size):
        crop = pil_img.crop(coords)

        if show_patches:
            cv2.imshow(
                "patch", cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(1)

        x = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        all_patches.append(emb.cpu().numpy())
        all_coords.append(coords)

    if not all_patches:
        return []

    vectors = np.vstack(all_patches).astype("float32")

    # Compare each patch embedding with each class database
    for class_name, class_db in stacked_databases.items():
        sims = cosine_similarity(vectors, class_db)
        max_sim = sims.max(axis=1)

        detections.extend(
            Detection(
                class_name=class_name,
                similarity=float(sim),
                coords=all_coords[idx],
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


def main():
    # Build the databases (just like your SIFT version)
    stacked_databases = {
        "R": build_features("./datasets/r/"),
        "RR": build_features("./datasets/rr/"),
        # Add more classes as needed, e.g.
        # "A": build_features("./datasets/a/"),
    }

    # Save them
    for name, base in stacked_databases.items():
        dump_database(f"database-{name}.npz", base)

    # Prepare output folder
    outpath = Path("datasets/test/v2-CLIP")
    outpath.mkdir(parents=True, exist_ok=True)

    # Run inference for each test image
    for file in Path("datasets/test-new").glob("*.png"):
        image = cv2.imread(str(file))
        detections = detect_objects(image, stacked_databases)
        save_detections(detections)
        detections = load_detections()
        annotated_raw = visualize_detections(image, detections)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_and_save("raw-", timestamp, annotated_raw)
        nma = non_maximum_averaging(detections)
        annotated_nma = visualize_detections(image, nma)
        plot_and_save("nma-", timestamp, annotated_nma)

    cv2.destroyAllWindows()


def plot_and_save(name, timestamp, image):
    cv2.imwrite(f"{name}{timestamp}.png", image)
    cv2.imshow("output", image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
