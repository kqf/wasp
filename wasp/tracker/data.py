import json
from dataclasses import dataclass, field
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
    label: Optional[int] = -1
    bbox: Optional[Bbox] = field(default_factory=lambda: Bbox(0, 0, 0, 0))

    def to_tuple(self) -> Optional[tuple[int, int, int, int]]:
        if self.bbox is None:
            return None
        x, y, w, h = map(int, self.bbox.to_tuple())
        return x, y, w, h


def read_data(path: str) -> list[Annotation]:
    with open(path, "r") as file:
        data = json.load(file)

    read = partial(Annotation.from_dict, infer_missing=False)  # type: ignore
    return [read(i) for i in data["images"]]
