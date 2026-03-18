import cv2
import torch
from torchvision.models.detection import retinanet_mobilenet_v3_large_fpn_v2
from torchvision.transforms import functional as F


def center_crop(image, size=120):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = size // 2
    return image[cy - half : cy + half, cx - half : cx + half]


def to_blob(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return F.to_tensor(image)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, input_tensor):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        with torch.no_grad():
            model([input_tensor])
    return sum(e.flops for e in prof.key_averages() if e.flops is not None)


def estimate_flops_naive(model, input_tensor):
    flops = 0

    def conv_hook(self, input, output):
        nonlocal flops
        x = input[0]
        batch_size, _, h, w = x.shape
        out_channels, _, kh, kw = self.weight.shape
        flops += batch_size * out_channels * h * w * kh * kw

    def linear_hook(self, input, output):
        nonlocal flops
        x = input[0]
        batch_size = x.shape[0]
        flops += batch_size * self.in_features * self.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model([input_tensor])

    for h in hooks:
        h.remove()

    return flops


def main():
    image = cv2.imread("image.jpg")
    crop = center_crop(image, 120)
    tensor = to_blob(crop)
    model = retinanet_mobilenet_v3_large_fpn_v2(weights="DEFAULT")
    model.eval()
    print(f"Total parameters: {count_parameters(model)}")
    print(f"Total flops: {estimate_flops(model, tensor)}")
    print(f"Total flops: {estimate_flops_naive(model, tensor.unsqueeze(0))}")
    with torch.no_grad():
        outputs = model([tensor])
    print(outputs)


if __name__ == "__main__":
    main()
