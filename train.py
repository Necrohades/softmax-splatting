import argparse
import datetime
import logging
import sys

import more_itertools
import numpy as np
import pathlib
import PIL.Image
import torch
import torch.nn
import torch.autograd
import torch.utils.tensorboard as tb
import tqdm

import dataset_triplet
import softmax_basic

DATETIME = datetime.datetime.now()
DATETIME_STR = DATETIME.strftime("%Y%m%d-%H%M%S")


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
    parser.add_argument("-s", "--state_dict", type=str, required=False, help="load state dict in folder location")
    parser.add_argument("-c", "--chunk_size", type=int, default=100)
    parser.add_argument("-V", "--validation_iterations", type=int, required=False)
    return parser.parse_args()


def _validate(network: torch.nn.Module,
              validation_loader: torch.utils.data.DataLoader,
              validation_iterations: int = None,
              *,
              progress_bar: bool = True,
              desc: str = None) -> tuple[float, float]:
    """
    Perform a simple validation loop over the validation data loader.
    :param network: Model to be evaluated.
    :param validation_loader: Validation data loader.
    :param validation_iterations: Number of total iterations. Defaults to the total length of the validation data loader.
    :param progress_bar: Whether to show a progress bar during the validation loop.
    :param desc: Description label of the progress bar if it is set to true.
    :return: A tuple containing the arithmetic mean of the L1 losses and the arithmetic mean of the MSE losses over all
        the validation data.
    """

    if validation_iterations is None:
        validation_iterations = len(validation_loader)

    validation_iter = zip(range(validation_iterations), validation_loader)
    if progress_bar:
        validation_iter = tqdm.tqdm(validation_iter, total=validation_iterations, desc=desc,
                                    ascii=True, dynamic_ncols=True, leave=True)

    validation_l1 = 0
    validation_mse = 0

    for _, (images, gt) in validation_iter:
        images = torch.autograd.Variable(
            torch.cat([image.view(*image.shape, -1) for image in map(preprocess, images)], dim=4)).cuda()
        gt = preprocess(gt).cuda()
        output = network(images)
        validation_l1 += torch.nn.functional.l1_loss(output, gt).item()
        validation_mse += torch.nn.functional.mse_loss(output, gt).item()

    return validation_l1 / validation_iterations, validation_mse / validation_iterations


def train(network: torch.nn.Module,
          dataset_folder_path: str,
          *,
          batch_size: int = 1,
          epochs: int = 1,
          chunk_size: int = 1000,
          learning_rate: float = 1e-4,
          progress_bar: bool = True,
          do_checkpoints: bool = True,
          checkpoint_dir: str = None,
          save_images: bool = True,
          save_dir: str = None,
          validation_iterations: int = None) -> None:

    logger = logging.getLogger(__name__)
    logger.info(f"Starting train iteration. Number of epochs: {epochs}; chunk size: {chunk_size}")

    train_data = dataset_triplet.Dataset(dataset_folder_path, split="train")
    validation_data = dataset_triplet.Dataset(dataset_folder_path, split="validation")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    iteration_number = 0  # counter for the number of training iterations so far

    if do_checkpoints:
        if checkpoint_dir is None:
            checkpoint_dir = "checkpoints/"
        checkpoint_path = pathlib.Path(checkpoint_dir)

    if save_images:
        if save_dir is None:
            save_dir = f"images/{DATETIME_STR}"  # default image save location
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)  # create save directory if it doesn't exist
        save_path = pathlib.Path(save_dir)

    with tb.SummaryWriter(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) as summary_writer:
        for n_epoch in range(epochs):
            train_iter = iter(train_loader)
            if progress_bar:  # decorate train data iterator with tqdm to show a progress bar
                train_iter = tqdm.tqdm(train_iter, desc=f"Epoch {n_epoch+1}/{epochs}", ascii=True, dynamic_ncols=True, leave=True)
            for i, chunk in enumerate(more_itertools.ichunked(train_iter, chunk_size)):
                if progress_bar:
                    chunk = tqdm.tqdm(chunk, total=chunk_size, ascii=True, dynamic_ncols=True, leave=True)
                for images, gt in chunk:
                    images = torch.autograd.Variable(torch.cat([image.view(*image.shape, -1) for image in map(preprocess, images)], dim=4)).cuda()
                    gt = preprocess(gt).cuda()

                    optimizer.zero_grad()
                    output = network(images)
                    loss = loss_function(output, gt)
                    l1_loss = torch.nn.functional.l1_loss(output, gt)
                    loss.backward()
                    optimizer.step()

                    summary_writer.add_scalar("Train/MSE", loss.item(), iteration_number)
                    summary_writer.add_scalar("Train/L1", l1_loss.item()), iteration_number
                    iteration_number += 1

                # periodically save the model parameters
                if do_checkpoints:
                    torch.save(network.state_dict(), checkpoint_path / f"{DATETIME_STR}-{i:03d}")

                # periodically save generated images
                if save_images:
                    to_image(output).save(save_path / f"{i:03d}_out.png")
                    to_image(gt).save(save_path / f"{i:03d}_gt.png")

                # validate the model after each chunk
                logger.info(f"Starting validation loop for chunk {i} of epoch {n_epoch}")
                validation_l1, validation_mse = _validate(network, validation_loader, validation_iterations,
                                                          progress_bar=progress_bar, desc=f"Chunk {i} validation")
                logger.info(f"c{i}e{n_epoch} validation results: MSE - {validation_mse}; L1 - {validation_l1}")
                summary_writer.add_scalar("Validation/MSE", validation_mse, i)
                summary_writer.add_scalar("Validation/L1", validation_l1, i)


def test(network: torch.nn.Module,
         dataset_folder_path: str,
         *,
         batch_size: int = 1,
         progress_bar: bool = True) -> None:

    test_data = dataset_triplet.Dataset(dataset_folder_path, split="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if progress_bar:
        test_loader = tqdm.tqdm(test_loader, desc="Test", ascii=True, dynamic_ncols=True, leave=True)

    with tb.SummaryWriter(log_dir="logs/test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) as summary_writer:
        for i, (images, gt) in enumerate(test_loader):
            images = torch.autograd.Variable(torch.cat([image.view(*image.shape, -1) for image in map(preprocess, images)], dim=4)).cuda()
            gt = preprocess(gt).cuda()
            output = network(images)

            mse_loss = torch.nn.functional.mse_loss(output, gt)
            l1_loss = torch.nn.functional.l1_loss(output, gt)

            summary_writer.add_scalar("Test/MSE", mse_loss.item(), i)
            summary_writer.add_scalar("Test/L1", l1_loss.item(), i)


def main():
    args = parse_args()

    if args.verbose:
        print("Verbose is set to true", file=sys.stderr)
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    net = softmax_basic.Model().cuda()
    if args.state_dict is not None:
        net.load_state_dict(torch.load(args.state_dict))

    if args.test:
        test(net, args.path, batch_size=args.batch_size)
    else:
        train(net, args.path, batch_size=args.batch_size, chunk_size=args.chunk_size, validation_iterations=args.validation_iterations)


if __name__ == '__main__':
    main()
