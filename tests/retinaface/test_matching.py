import torch

from wasp.retinaface.matching import match


def test_match(
    threshold=0.5,
    box_gt=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
    priors=torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
    variances=None,
    labels_gt=torch.tensor([1, 2]),
    landmarks_gt=torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        ]
    ),
    batch_id=0,
):
    variances = variances or [0.1, 0.1, 0.2, 0.2]
    box_t = torch.zeros(1, priors.size(0), 4)
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
