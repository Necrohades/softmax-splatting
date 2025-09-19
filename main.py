import dataset_triplet
import softmax_basic

import torch
import torch.nn
import torchvision.transforms
from torch.autograd import Variable

import logging
import sys


class AffineFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sin = torch.nn.Parameter(requires_grad=True)
        self.cos = torch.nn.Parameter(requires_grad=False)
    
    def forward(self, tenOne: torch.Tensor, tenTwo: torch.Tensor):
        intWidth = tenOne.shape[3] and tenTwo.shape[3]
        intHeight = tenOne.shape[2] and tenTwo.shape[2]

        tenOne = self.netExtractor(tenOne)

import PIL.Image
import numpy as np


def preprocess(image: torch.Tensor) -> torch.Tensor:
    batch_size, height, width, channels = image.shape
    assert channels == 3
    preprocessed = image.cuda().permute((0, 3, 1, 2))  # .view(batch_size, channels, height, width)
    pad = [0, width & 1, 0, height & 1]
    if (width | height) & 1:
        preprocessed = torch.nn.functional.pad(input=preprocessed, pad=pad, mode="replicate")
    return preprocessed


def to_image(image: torch.Tensor, index: int = 0) -> PIL.Image.Image:
    array = image.clip(0.0, 1.0).permute(0, 2, 3, 1)[index].numpy(force=True) * 255.0
    return PIL.Image.fromarray(array.astype(np.uint8))


def main():
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    # raw input: 448x256
    inputWidth = 448
    inputHeight = 256
    input_size = inputWidth * inputHeight * 3
    batch_size = 1
    learning_rate = 1e-4

    train_data = dataset_triplet.Dataset("/home/tonifuentes/Pictures/archive/vimeo_triplet/", split="train")
    test_data  = dataset_triplet.Dataset("/home/tonifuentes/Pictures/archive/vimeo_triplet/", split="test")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    net = softmax_basic.Model()
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    netNetwork = softmax_basic.Model().cuda()

    for _ in range(1):
        for i, (images, gt) in enumerate(train_loader, 1):
            print(images[0].shape)
            transforms = torchvision.transforms.ToTensor()
            # images = [image.numpy().transpose(0, 3, 1, 2)[:, ::-1, :, :].astype(np.float32) for image in images]
            # images = (transforms(image) for image in images)
            images = Variable(torch.cat([image.view(batch_size, 3, inputHeight, inputWidth, -1) for image in images], dim=4)).cuda()
            # gt = gt.numpy().transpose(0, 3, 1, 2)[:, ::-1, :, :]
            # gt = torch.from_numpy(gt.copy()).cuda()
            gt = preprocess(gt)
            print(gt.shape)

            optimizer.zero_grad()
            output = netNetwork(images)
            output = output.view(*output.shape[1:])

            loss = loss_function(output, gt)
            loss.backward()
            optimizer.step()

            if not i % 1:
                print(f"step: {i}, loss: {loss.item()}")
                # PIL.Image.fromarray((output.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(f"film_out{i//100}.png")
                # PIL.Image.fromarray((gt.clip(0.0, 1.0).view(*gt.shape[1:]).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(f"film_gt{i//100}.png")
                # PIL.Image.fromarray((gt.clip(0.0, 1.0).view(*gt.shape[1:]).numpy(force=True) * 255.0).astype(np.uint8)).save(f"film_gt{i//100}.png")
                to_image(output).save(f"film_out{i//100}.png")
                to_image(gt).save(f"film_gt{i//100}.png")



if __name__ == "__main__":
    main()

