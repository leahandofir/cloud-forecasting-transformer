# TODO: load checkpoints from checkpoints directory

import argparse
import os
import torch

from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class CuboidIMSInference():
    def __init__(self, ckpt_path):
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            # TODO: we can load all model cfg from the checkpoint and then load the weights from checkpoint
            model = CuboidTransformerModel()
            self.model = model.load_from_checkpoint(ckpt_path)

    def _preprocess(self, data):
        pass

    def _inference(self, data):
        pass

    def _postprocess(self, data):
        pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default=None, type=str)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    ims_inference = CuboidIMSInference(ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
