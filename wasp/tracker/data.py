import json
from dataclasses import dataclass
from functools import partial
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Bbox:
    left: float
    top: float
    width: float
    height: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.width, self.height


@dataclass_json
@dataclass
class Annotation:
    file: str
    label: int
    bbox: Optional[Bbox]

    def to_tuple(self) -> Optional[tuple[int, int, int, int]]:
        if self.bbox is None:
            return None
        x, y, w, h = map(int, self.bbox.to_tuple())
        return x, y, w, h


def read_data(path: str) -> list[Annotation]:
    with open(path, "r") as file:
        data = json.load(file)

    read = partial(Annotation.from_dict, infer_missing=True)  # type: ignore
    return [read(i) for i in data["images"]]
