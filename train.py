import argparse
import datetime
import itertools
import random

import more_itertools
import numpy as np
import PIL.Image
import torch
import torch.nn
import torch.autograd
import torch.utils.tensorboard as tb
import tensorboard
import tqdm

import dataset_triplet
import softmax_basic


def preprocess(image: torch.Tensor) -> torch.Tensor:
    batch_size, height, width, channels = image.shape
    assert channels == 3
    preprocessed = image.cuda().permute((0, 3, 1, 2))  # .view(batch_size, channels, height, width)
    pad = [0, width & 1, 0, height & 1]
    if (width | height) & 1:
        preprocessed = torch.nn.functional.pad(input=preprocessed, pad=pad, mode="replicate")
    return preprocessed


def to_image(image: torch.Tensor, index: int = 0) -> PIL.Image.Image:
    array = image.clip(0.0, 1.0).permute((0, 2, 3, 1))[index].numpy(force=True) * 255.0
    return PIL.Image.fromarray(array.astype(np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("-p", "--path", required=False, default="/home/tonifuentes/Pictures/archive/vimeo_triplet/")
    parser.add_argument("-w", "--warp", nargs=2, required=False, help="location to store warped images.")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    learning_rate = 1e-4

    split = "test" if args.test else "train"
    data = dataset_triplet.Dataset(args.path, split=split)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=not args.test)

    net = softmax_basic.Model().cuda()
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    verbose_batch_size = 100
    iteration_number = 0
    with tb.SummaryWriter(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) as summary_writer:
        for n_epoch in range(args.epochs):
            for i, batch in enumerate(more_itertools.ichunked(tqdm.tqdm(loader, desc=f"Epoch {n_epoch+1}/{args.epochs}", ascii=True, dynamic_ncols=True, leave=False), verbose_batch_size)):
                for images, gt in tqdm.tqdm(batch, total=verbose_batch_size, ascii=True, dynamic_ncols=True, leave=False):
                    images = torch.autograd.Variable(torch.cat([image.view(*image.shape, -1) for image in map(preprocess, images)], dim=4)).cuda()
                    gt = preprocess(gt).cuda()
                    optimizer.zero_grad()
                    output = net(images)
                    loss = loss_function(output, gt)
                    loss.backward()
                    optimizer.step()

                    if True:
                        summary_writer.add_scalar(f"Loss/random", random.random(), iteration_number)
                        summary_writer.add_scalar(f"Loss/{split}", float(loss.item()), iteration_number)
                    iteration_number += 1
                print(f"checkpoint: {i}, loss: {loss.item()}")
                to_image(output).save(f"film_out{i}.png")
                to_image(gt).save(f"film_gt{i}.png")
                summary_writer.flush()
                torch.save(net.state_dict(), f"checkpoints/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt")


if __name__ == '__main__':
    main()
