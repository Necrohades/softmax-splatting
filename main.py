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

import PIL
import numpy as np

def main():
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    # raw input: 448x256
    inputWidth = 448
    inputHeight = 256
    input_size = inputWidth * inputHeight * 3
    batch_size = 5
    learning_rate = 1e-3

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
            gt = gt.numpy().transpose(0, 3, 1, 2)[:, ::-1, :, :]
            gt = torch.from_numpy(gt.copy()).cuda()

            optimizer.zero_grad()
            output = netNetwork(images)
            output = output.view(batch_size, *output.shape[2:])

            loss = loss_function(output, gt)
            loss.backward()
            optimizer.step()

            if not i % 1:
                print(f"step: {i}, loss: {loss.item()}")
                PIL.Image.fromarray((output.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(f"film_out{i//100}.png")
                PIL.Image.fromarray((gt.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)).save(f"film_gt{i//100}.png")


if __name__ == "__main__":
    main()

