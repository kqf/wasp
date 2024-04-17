import pytest
import torch

from wasp.retinaface.encode import decode, decode_landm, encode, encode_landm


@pytest.fixture
def matched(encode):
    if encode == encode_landm:
        return torch.arange(20).reshape(2, 10).float()
    # Define matched data
    return torch.tensor([[1, 1, 3, 3, 5] * 2, [2, 2, 4, 4.0, 4] * 2])


@pytest.fixture
def priors(encode):
    if encode == encode_landm:
        return torch.arange(20).reshape(2, 10).float()
    # Define priors data
    return torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3.0]])


@pytest.fixture
def variances():
    # Define variances data
    return [0.1, 0.2]


@pytest.fixture
def encoded(encode, matched, priors, variances):
    # Call the encode function
    return encode(matched, priors, variances)


@pytest.fixture
def decoded(decode, encoded, priors, variances):
    # Call the decode function
    return decode(encoded, priors, variances)


@pytest.mark.parametrize(
    "encode, decode",
    [
        (encode, decode),
        (encode_landm, decode_landm),
    ],
)
def test_encodes_decodes(decoded, matched):
    # Assert the decoded matches the original matched data
    assert torch.allclose(decoded, matched, atol=1e-4)
