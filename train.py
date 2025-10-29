import argparse
import datetime
import logging
import sys

import more_itertools
import numpy as np
import pathlib
import PIL.Image
import rich.progress
import torch
import torch.nn
import torch.autograd
import torch.utils.tensorboard as tb
import torcheval.metrics
import tqdm
from tensorflow.python.ops.gen_summary_ops import summary_writer

from typing import Any, Tuple

import dataset_triplet
import softmax_basic
import softsplat

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


def detect_constant_image(image: torch.Tensor) -> bool:
    """Checks whether all values in a given array are equal."""
    return image.max().item() == image.min().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("-p", "--path", required=False, default="/home/tonifuentes/Pictures/archive/vimeo_triplet/")
    parser.add_argument("-w", "--warp", nargs=2, required=False, help="location to store warped images.")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-s", "--state_dict", type=str, required=False, help="load state dict in folder location")
    parser.add_argument("-r", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-C", "--checkpoint_period", type=int, default=1000)
    parser.add_argument("-V", "--validation_period", type=int, default=100)
    parser.add_argument("-I", "--image_generation_period", type=int, default=1000)
    parser.add_argument("--validation_iterations", type=int, required=False, default=10)
    parser.add_argument("-d", "--disable_generate_files", action="store_true", default=None)
    return parser.parse_args()


def _validate(network: torch.nn.Module,
              validation_loader: torch.utils.data.DataLoader,
              validation_iterations: int = None,
              *,
              description: str = None) -> Tuple[float, float, float]:
    """
    Perform a simple validation loop over the validation data loader.
    :param network: Model to be evaluated.
    :param validation_loader: Validation data loader.
    :param validation_iterations: Number of total iterations. Defaults to the total length of the validation data loader.
    :param description: Description label of the progress bar.
    :return: A tuple containing the arithmetic mean of the L1 losses and the arithmetic mean of the MSE losses over all
        the validation data.
    """

    if validation_iterations is None:
        validation_iterations = len(validation_loader)

    validation_iter = zip(range(validation_iterations), validation_loader)

    validation_l1 = 0
    validation_mse = 0
    psnr = 0

    with rich.progress.Progress() as progress:
        validation_task = progress.add_task(description, total=validation_iterations)
        for _, (images, gt) in validation_iter:
            images = torch.autograd.Variable(
                torch.cat([image.view(*image.shape, -1) for image in map(preprocess, images)], dim=4)).cuda()
            gt = preprocess(gt).cuda()
            output = network(images)
            validation_l1 += torch.nn.functional.l1_loss(output, gt).item()
            validation_mse += torch.nn.functional.mse_loss(output, gt).item()
            metric = torcheval.metrics.PeakSignalNoiseRatio()
            metric.update(gt, output)
            psnr += metric.compute().item()
            if progress is not None:
                progress.update(validation_task, advance=1)

    return psnr / validation_iterations, validation_l1 / validation_iterations, validation_mse / validation_iterations


def warp(network: softmax_basic.Model, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ten_one = images[:, :, :, :, 0]
    ten_two = images[:, :, :, :, 1]

    with torch.set_grad_enabled(False):
        ten_stats = [ten_one, ten_two]
        ten_mean = sum([ten_in.mean([1, 2, 3], True) for ten_in in ten_stats]) / len(ten_stats)
        ten_std = (sum(ten_in.std([1, 2, 3], unbiased=False, keepdim=True).square() + (ten_mean - ten_in.mean([1, 2, 3], True)).square()
                       for ten_in in ten_stats) / len(ten_stats)).sqrt()
        ten_one = ((ten_one - ten_mean) / (ten_std + 1e-7)).detach()
        ten_two = ((ten_two - ten_mean) / (ten_std + 1e-7)).detach()

    ten_forward, ten_backward = network.netFlow(ten_one, ten_two)
    ten_enc_one = network.netSynthesis.netEncode(ten_one)
    ten_enc_two = network.netSynthesis.netEncode(ten_two)

    ten_metric_one = network.netSynthesis.netSoftmetric(ten_enc_one, ten_enc_two, ten_forward)
    ten_metric_two = network.netSynthesis.netSoftmetric(ten_enc_two, ten_enc_one, ten_backward)

    ten_forward = ten_forward * .5
    ten_backward = ten_backward * .5

    ten_warp1 = softsplat.softsplat(ten_one, ten_forward, ten_metric_one.neg().clip(-20, 20.0), "soft")
    ten_warp2 = softsplat.softsplat(ten_two, ten_backward, ten_metric_two.neg().clip(-20, 20.0), "soft")
    ten_warp1 = (ten_warp1 * ten_std) + ten_mean
    ten_warp2 = (ten_warp2 * ten_std) + ten_mean

    return ten_warp1, ten_warp2


def train(network: softmax_basic.Model,
          dataset_folder_path: str,
          *,
          batch_size: int = 1,
          epochs: int = 1,
          initial_learning_rate: float = 1e-4,
          final_learning_rate: float = 1e-7,
          checkpoint_save_period: int = 1000,
          checkpoint_dir: str = None,
          image_save_period: int = 1000,
          save_dir: str = None,
          validation_period: int = 0,
          validation_iterations: int = 10,
          verbose: bool = False) -> None:

    logger = logging.getLogger(__name__)
    logger.info(f"Starting train iteration. Number of epochs: {epochs}")

    train_data = dataset_triplet.Dataset(dataset_folder_path, split="train")
    validation_data = dataset_triplet.Dataset(dataset_folder_path, split="validation")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    loss_function = torch.nn.MSELoss()
    iteration_number = 0  # counter for the number of training iterations so far

    checkpoint_path: pathlib.Path | None = None
    if checkpoint_save_period:
        if checkpoint_dir is None:
            checkpoint_dir = f"checkpoints/{DATETIME_STR}"
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=False)
        checkpoint_path = pathlib.Path(checkpoint_dir)

    save_path: pathlib.Path | None = None
    if image_save_period:
        if save_dir is None:
            save_dir = f"images/{DATETIME_STR}"  # default image save location
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=False)  # create save directory if it doesn't exist
        save_path = pathlib.Path(save_dir)

    summary_writer = tb.SummaryWriter(log_dir="logs/fit/" + DATETIME_STR)
    progress = rich.progress.Progress()
    progress.__enter__()
    last_checkpoint: pathlib.Path | None = None

    try:
        for n_epoch in range(epochs):
            lr = n_epoch / (epochs - 1) if epochs > 1 else 0
            lr = initial_learning_rate * (1 - lr) + final_learning_rate * lr
            progress.__exit__(None, None, None)
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
            logger.info(f"Epoch {n_epoch}. lr={lr}")
            progress = rich.progress.Progress()
            progress.__enter__()
            epoch_task = progress.add_task(f"Epoch {n_epoch}/{epochs}", total=len(train_loader))
            for n_batch, (images, gt) in enumerate(train_loader):
                images = torch.autograd.Variable(torch.cat([image.view(*image.shape, -1) for image in map(preprocess, images)], dim=4)).cuda()
                gt = preprocess(gt).cuda()
                optimizer.zero_grad()
                output = network(images)
                batch_str = f"e{n_epoch:03d}b{n_batch:05d}"
                ten_warp1, ten_warp2 = warp(network, images)

                # restore to previous checkpoint if all values in the image are constant
                if detect_constant_image(output):
                    logger.warning(f"Detected corrupted image in {batch_str}")
                    logger.info("Warping images...")
                    ten_warp1, ten_warp2 = warp(network, images)

                    # normalize generated tensors to improve resulting image quality
                    # ten_warp1 = (ten_warp1 + 1) / 4
                    # ten_warp2 = (ten_warp2 + 1) / 4

                    # save generated images
                    logger.info(f"saving image {save_path / f'{batch_str}_out.png'}")
                    to_image(output).save(save_path / f"{batch_str}_out.png")
                    logger.info(f"saving image {save_path / f'{batch_str}_gt.png'}")
                    to_image(gt).save(save_path / f"{batch_str}_gt.png")
                    path_w1 = save_path / f"{batch_str}_w1.png"
                    path_w2 = save_path / f"{batch_str}_w2.png"
                    logger.info(f"saving warped image {path_w1}")
                    to_image(ten_warp1).save(path_w1)
                    logger.info(f"saving warped image {path_w2}")
                    to_image(ten_warp2).save(path_w2)

                    if last_checkpoint is None:
                        logger.error("Cannot restore to previous checkpoint. Resetting model...")
                        network = softmax_basic.Model().cuda()
                    else:
                        logger.info(f"Restoring to checkpoint {last_checkpoint}")
                        network.load_state_dict(torch.load(last_checkpoint))
                    continue

                loss = loss_function(output, gt)
                loss.backward()
                optimizer.step()
                l1_loss = torch.nn.functional.l1_loss(output, gt)

                summary_writer.add_scalar("Train/MSE", loss.item(), iteration_number)
                summary_writer.add_scalar("Train/L1", l1_loss.item(), iteration_number)
                restore_progress = False

                # periodically save checkpoints
                if checkpoint_save_period > 0 and not n_batch % checkpoint_save_period:
                    if verbose and not restore_progress:  # close progress bar if printing logger info to stderr
                        progress.__exit__(None, None, None)
                        restore_progress = True
                    last_checkpoint = checkpoint_path / batch_str
                    logger.info(f"saving checkpoint {last_checkpoint}...")
                    torch.save(network.state_dict(), last_checkpoint)

                # periodically save generated images
                if image_save_period > 0 and not n_batch % image_save_period:
                    if verbose and not restore_progress:  # close progress bar if printing logger info to stderr
                        progress.__exit__(None, None, None)
                        restore_progress = True
                    logger.info(f"saving image {save_path / f'e{batch_str}_out.png'}...")
                    to_image(output).save(save_path / f"e{batch_str}_out.png")
                    logger.info(f"saving image {save_path / f'e{batch_str}_gt.png'}...")
                    to_image(gt).save(save_path / f"e{batch_str}_gt.png")

                if validation_period > 0 and iteration_number and not n_batch % validation_period:
                    if True or verbose and not restore_progress:
                        progress.__exit__(None, None, None)
                        restore_progress = True
                    psnr, validation_l1, validation_mse = _validate(network, validation_loader, validation_iterations, description=f"Batch {batch_str} validation")
                    logger.info(f"{batch_str} validation results: PSNR - {psnr}, MSE - {validation_mse}; L1 - {validation_l1}")
                    summary_writer.add_scalar("Validation/PSNR", psnr, iteration_number)
                    summary_writer.add_scalar("Validation/MSE", validation_mse, iteration_number)
                    summary_writer.add_scalar("Validation/L1", validation_l1, iteration_number)

                if restore_progress:
                    progress = rich.progress.Progress()
                    progress.__enter__()
                    epoch_task = progress.add_task(f"Epoch {n_epoch}/{epochs}", total=len(train_loader), completed=n_batch)

                iteration_number += 1
                progress.update(epoch_task, advance=1)
                progress.refresh()

    except Exception as e:
        progress.__exit__(type(e), e, e.__traceback__)
        raise e
    else:
        progress.__exit__(None, None, None)
    finally:
        summary_writer.close()


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
        print("Verbose is set to true", file=sys.stdout)
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    net = softmax_basic.Model().cuda()
    if args.state_dict is not None:
        net.load_state_dict(torch.load(args.state_dict))

    if args.test:
        test(net, args.path, batch_size=args.batch_size)
    else:
        train(
            net, args.path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_save_period=args.checkpoint_period,
            validation_period=args.validation_period,
            validation_iterations=args.validation_iterations,
            image_save_period=args.image_generation_period,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
