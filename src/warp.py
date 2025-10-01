import numpy
import PIL
import PIL.Image
import torch
from typing import Tuple

import run
import softsplat


def to_image(image: torch.Tensor, index: int = 0) -> PIL.Image.Image:
    array = image.clip(0.0, 1.0).permute((0, 2, 3, 1))[index].numpy(force=True) * 255.0
    return PIL.Image.fromarray(array[:, :, ::-1].astype(numpy.uint8))


def warp(network: run.Network, ten_one: torch.Tensor, ten_two: torch.Tensor, time: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.set_grad_enabled(False):
        ten_stats = [ten_one, ten_two]
        ten_mean = sum([ten_in.mean([1, 2, 3], True) for ten_in in ten_stats]) / len(ten_stats)
        ten_std = (sum(ten_in.std([1, 2, 3], unbiased=False, keepdim=True).square() + (ten_mean - ten_in.mean([1, 2, 3], True)).square()
                       for ten_in in ten_stats) / len(ten_stats)).sqrt()
        ten_one = ((ten_one - ten_mean) / (ten_std + 1e-7)).detach()
        ten_two = ((ten_two - ten_mean) / (ten_std + 1e-7)).detach()

    obj_flow = network.netFlow(ten_one, ten_two)
    ten_forward = obj_flow["tenForward"]
    ten_backward = obj_flow["tenBackward"]
    ten_enc_one = network.netSynthesis.netEncode(ten_one)
    ten_enc_two = network.netSynthesis.netEncode(ten_two)

    ten_metric_one = network.netSynthesis.netSoftmetric(ten_enc_one, ten_enc_two, ten_forward) * 2.0 * time
    ten_metric_two = network.netSynthesis.netSoftmetric(ten_enc_two, ten_enc_one, ten_backward) * 2.0 * (1 - time)

    ten_forward = ten_forward * time
    ten_backward = ten_backward * (1 - time)

    ten_warp1 = softsplat.softsplat(ten_one, ten_forward, ten_metric_one.neg().clip(-20, 20.0), "soft")
    ten_warp2 = softsplat.softsplat(ten_two, ten_backward, ten_metric_two.neg().clip(-20, 20.0), "soft")
    ten_warp1 = (ten_warp1 * ten_std) + ten_mean
    ten_warp2 = (ten_warp2 * ten_std) + ten_mean

    return ten_warp1, ten_warp2


# end

##########################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="warp")
    parser.add_argument("--one", type=str, required=True)
    parser.add_argument("--two", type=str, required=True)
    parser.add_argument("-t", "--time", type=float, default=0.5)
    parser.add_argument("--w1", type=str, default="w1.png")
    parser.add_argument("--w2", type=str, default="w2.png")
    args = parser.parse_args()

    ten_one = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(PIL.Image.open(args.one))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0
    ))
    ten_two = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(PIL.Image.open(args.two))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0
    ))

    if run.netNetwork is None:
        netNetwork = run.Network().cuda().train(False)

    ten_warp1, ten_warp2 = warp(run.netNetwork, ten_one, ten_two, args.time)

    to_image(ten_warp1).save(args.w1)
    to_image(ten_warp2).save(args.w2)
