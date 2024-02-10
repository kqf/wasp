import json
import pathlib

import pytest

from wasp.retinaface.train import main


@pytest.fixture
def annotations(tmp_path) -> pathlib.Path:
    example = [
        {
            "file_name": "tests/assets/lenna.png",
            "annotations": [
                {
                    "bbox": [449, 330, 571, 720],
                    "landmarks": [
                        [488.906, 373.643],
                        [542.089, 376.442],
                        [515.031, 412.83],
                        [485.174, 425.893],
                        [538.357, 431.491],
                    ],
                }
            ],
        },
    ]
    ofile = tmp_path / "annotations.json"
    with open(ofile, "w") as f:
        json.dump(example, f, indent=2)
    return ofile


@pytest.mark.timeout(3600)
def test_main(annotations):
    main(
        train_labels=str(annotations),
        valid_labels=str(annotations),
        resolution=(256, 256),
    )
