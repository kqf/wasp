import json
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json


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
