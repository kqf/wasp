import json
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./wider_face_split/wider_face_train_bbx_gt.txt"),
)
@click.option(
    "--ofile",
    type=click.Path(exists=False, path_type=Path),
    default="wider.json",
)
def main(dataset, ofile):
    result = []
    temp = {}

    valid_idx = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])
    with open(dataset) as f:
        for line_id, line in enumerate(f.readlines()):
            if line[0] == "#":
                if line_id != 0:
                    result += [temp]

                temp = {
                    "file_name": line.replace("#", "").strip(),
                    "annotations": [],
                }

            else:
                points = line.strip().split()

                x_min = int(points[0])
                y_min = int(points[1])
                x_max = int(points[2]) + x_min
                y_max = int(points[3]) + y_min

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)

                x_max = max(x_min + 1, x_max)
                y_max = max(y_min + 1, y_max)

                landmarks = np.array([float(x) for x in points[4:]])

                if landmarks.size > 0:
                    landmarks = landmarks[valid_idx].reshape(-1, 2).tolist()
                else:
                    landmarks = []

                temp["annotations"] += [
                    {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "landmarks": landmarks,
                    }
                ]

        result += [temp]

    with open(ofile, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
