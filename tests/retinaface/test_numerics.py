import torch

from wasp.retinaface.loss import log_sum_exp


def test_numerics():
    batch = torch.arange(10).reshape(-1, 2) / 10.0
    batch[:, 0] = 0.5
    batch[:, 1] = 0.5
    batch[-2:, 0] = 0.8
    batch[-2:, 1] = 0.2
    batch[-1:, 0] = 0.9
    batch[-1:, 1] = 0.1
    print()
    print(batch)
    print(log_sum_exp(batch))
