import os

import pytest


def model_exists():
    file_to_check = "models/det_10g.onnx"
    return os.path.exists(file_to_check)


@pytest.mark.skipif(
    not model_exists(),
    reason="File doesn't exists, skipping the test",
)
def test_srfd_inferencd():
    print("HERE")
