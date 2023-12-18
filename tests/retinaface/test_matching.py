import torch

from wasp.retinaface.matching import match


def test_match(
    threshold=0.5,
    priors=torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
    box_gt=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
    labels_gt=torch.tensor([1, 2]),
    landmarks_gt=torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        ]
    ),
    batch_id=0,
    variances=None,
):
    variances = variances or [0.1, 0.1, 0.2, 0.2]
    box_t = torch.zeros(1, priors.shape[0], 4)
    label_t = torch.zeros(1, 2)
    landmarks_t = torch.zeros(1, 2, 10)
    match(
        threshold=threshold,
        box_gt=box_gt,
        priors=priors,
        variances=variances,
        labels_gt=labels_gt,
        landmarks_gt=landmarks_gt,
        box_t=box_t,
        label_t=label_t,
        landmarks_t=landmarks_t,
        batch_id=batch_id,
    )
    torch.testing.assert_close(label_t, torch.Tensor([[1.0, 0.0]]))
    print(box_t.cpu().numpy().__repr__())
    torch.testing.assert_close(
        box_t,
        torch.Tensor(
            [
                [
                    [0.000000e00, 0.0000000e00, -6.9314704e00, -9.1629066e00],
                    [0.000000e00, 6.6227386e-07, -1.3862944e01, -1.5040774e01],
                ]
            ]
        ),
    )
    torch.testing.assert_close(
        landmarks_t[0],
        torch.Tensor(
            [
                [
                    -2.4999998,
                    -2.0000002,
                    2.5,
                    1.9999999,
                    7.5,
                    6.0,
                    12.499999,
                    10.0,
                    17.499998,
                    14.0,
                ],
                [
                    -5.0,
                    -4.444444,
                    -2.5,
                    -2.222222,
                    0.0,
                    0.0,
                    2.4999998,
                    2.222222,
                    4.9999995,
                    4.444445,
                ],
            ]
        ),
    )
