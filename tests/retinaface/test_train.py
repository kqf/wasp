import json

import pytest

from wasp.retinaface.train import Paths, main

EXAMPLE = [
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


@pytest.fixture
def annotations(tmp_path):
    ofile = tmp_path / annotations.json
    with open(ofile, "w") as f:
        json.dump(EXAMPLE, f, indent=2)
    return ofile


def test_main(tmp_path):
    main(
        paths=Paths(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
        )
    )
