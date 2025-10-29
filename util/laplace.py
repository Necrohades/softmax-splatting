import itertools
import torch
import skimage

from typing import Iterator


__FACTORIAL = [1]  # memoized factorial numbers


def gauss_kernel(size: int, channels=3) -> torch.Tensor:
    while len(__FACTORIAL) < size:  # extend memoized factorial numbers as needed
        __FACTORIAL.append(__FACTORIAL[-1] * len(__FACTORIAL))
    binom_coefficients = [__FACTORIAL[size-1] / (__FACTORIAL[i] * __FACTORIAL[size-i-1]) for i in range(size)]
    a = torch.Tensor([binom_coefficients])  # row vector with binomial coefficient items
    b = torch.Tensor([binom_coefficients]).reshape((-1, 1)) # colum vector
    # the sum of all the numbers in a * b is 2 ** (2 * (size - 1))
    total_sum = 1 << (size - 1 << 1)
    return (a * b / total_sum).repeat(channels, 1, 1, 1)


def downsample(image, scale_factor=2) -> torch.Tensor:
    return image[:, :, ::scale_factor, ::scale_factor]


def upsample(image, scale_factor=2) -> torch.Tensor:
    new_shape = (image.shape[0], image.shape[1], image.shape[2] * scale_factor, image.shape[3] * scale_factor)
    return torch.nn.functional.interpolate(image, scale_factor=scale_factor, mode="bilinear")


def blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    image = torch.nn.functional.pad(image, (2, 2, 2, 2), mode="reflect")
    return torch.nn.functional.conv2d(image, kernel, groups=image.shape[1])


def laplacian_pyramid(image: torch.Tensor, kernel: torch.Tensor, levels=3, scale_factor=2) -> Iterator[torch.Tensor]:
    for level in range(levels):
        downsampled = downsample(blur(image, kernel), scale_factor)
        upsampled = upsample(downsampled, scale_factor)
        yield image - upsampled
        image = downsampled


def lap_loss(input, target) -> torch.Tensor:
    pyramid_input = skimage.transform.pyramid_laplacian(input.cpu())
    pyramid_target = skimage.transform.pyramid_laplacian(target.cpu())
    return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyramid_input, pyramid_target))


class LaplacianLoss(torch.nn.Module):
    def __init__(self, levels=3, channels=3, kernel_size=5, device=torch.device("cpu")) -> None:
        super().__init__()
        self.levels = levels
        self.channels = channels
        self.kernel = gauss_kernel(kernel_size, channels).to(device)
        self.device = device

    def _blur(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def _pyramid(self, image: torch.Tensor) -> Iterator[torch.Tensor]:
        for _ in range(self.levels):
            downsampled = downsample(self._blur(image))
            upsampled = upsample(downsampled)
            yield image - upsampled
            image = downsampled

    def forward(self, input, target) -> torch.Tensor:
        pyramid_input = laplacian_pyramid(input, self.kernel, self.levels)
        pyramid_target = laplacian_pyramid(target, self.kernel, self.levels)
        return sum(itertools.starmap(torch.nn.functional.l1_loss, zip(pyramid_input, pyramid_target)))
