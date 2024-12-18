import json
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Segment:
    start_frame: int
    stop_frame: int
    last_frame: int
    bbox: tuple[float, float, float, float]
    tracker: str
    name: str

    def within(self, frame_count):
        return self.start_frame <= frame_count < self.stop_frame


def save_segments(segments: dict[str, Segment], filename: str) -> None:
    data = {k: s.to_dict() for k, s in segments.items()}  # type: ignore
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_segments(filename: str) -> dict[str, Segment]:
    with open(filename, "r") as json_file:
        data = json.load(json_file)

    return {k: Segment.from_dict(v) for k, v in data.items()}  # type: ignore
