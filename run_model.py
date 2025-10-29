import argparse

import PIL.Image
import numpy as np
import torch

import softmax_basic
import train
import warp


def parse_args():
    parser = argparse.ArgumentParser('run_model')
    parser.add_argument('one', type=str)
    parser.add_argument('two', type=str)
    parser.add_argument('-o', '--out', type=str, default='out.png')
    parser.add_argument('-f', '--flow', type=str,
                        help="path to the pre-trained flow network state dict")
    parser.add_argument('-s', '--synthesis', type=str,
                        help="path to the pre-trained synthesis network state dict")
    return parser.parse_args()


def main(one: str, two: str, out: str, *,
         w1_path: str = None, w2_path: str = None, flow_path: str = None, synthesis_path: str = None):
    model = softmax_basic.Model().cuda().train(False)
    if flow_path is not None:
        model.netFlow.load_state_dict(torch.load(flow_path))
    if synthesis_path is not None:
        model.netFlow.load_state_dict(torch.load(synthesis_path))

    dot_index = out.rindex('.')
    if w1_path is None:
        w1_path = out[:dot_index] + '_w1' + out[dot_index:]
    if w2_path is None:
        w2_path = out[:dot_index] + '_w2' + out[dot_index:]

    ten_one = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(one)).astype(np.float32) / 255.0))
    ten_two = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(two)).astype(np.float32) / 255.0))
    print(ten_one.shape)
    ten_one = train.preprocess(ten_one.view(1, *ten_one.shape))
    ten_two = train.preprocess(ten_two.view(1, *ten_two.shape))
    w1, w2 = warp.warp(model, ten_one, ten_two)
    train.to_image(w1).save(w1_path)
    train.to_image(w2).save(w2_path)

    in_img = torch.cat([image.view(*image.shape, -1) for image in [ten_one, ten_two]], dim=4).cuda()
    out_img = model(in_img)
    train.to_image(out_img).save(out)


if __name__ == '__main__':
    args = parse_args()
    main(args.one, args.two, args.out, flow_path=args.flow, synthesis_path=args.synthesis)
