import pytest
import torch

from wasp.retinaface.encode import decode, encode


@pytest.fixture
def matched():
    # Define matched data
    return torch.tensor([[1, 1, 3, 3], [2, 2, 4, 4]], dtype=torch.float32)


@pytest.fixture
def priors():
    # Define priors data
    return torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32)


@pytest.fixture
def variances():
    # Define variances data
    return [0.1, 0.2]


@pytest.fixture
def encoded(matched, priors, variances):
    # Call the encode function
    return encode(matched, priors, variances)


@pytest.fixture
def decoded(encoded, priors, variances):
    # Call the decode function
    return decode(encoded, priors, variances)


def test_encodes_decodes(decoded, matched):
    # Assert the decoded matches the original matched data
    assert torch.allclose(decoded, matched, atol=1e-4)
