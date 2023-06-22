import argparse
import os
import sys
import png
import numpy as np
from datetime import datetime, timedelta

import torch

from src.earthformer.utils.ims.load_model import load_model
from src.earthformer.config import cfg
from src.earthformer.datasets.ims.ims_dataset import IMSPreprocess
from src.earthformer.visualization.ims.ims_visualize import IMSVisualize

START_TIME_FORMAT = "%Y%m%d%H%M"
IMAGE_NAME_FORMAT = "%Y%m%d%H%M"
SUPPORTED_FORMATS = ('png', )

pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir


class CuboidIMSInference:
    def __init__(self, ckpt_name, # TODO: add types!
                 data_dir,
                 start_time,
                 img_format,
                 output_dir,
                 fs,
                 figsize,
                 plot_stride,
                 left,
                 top,
                 width,
                 height):

        self.ckpt_name = ckpt_name

        # load model
        ckpt_path = os.path.join(pretrained_checkpoints_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
        else:
            print("The ckpt file path does not exist!")
            sys.exit()

        model_cfg = checkpoint["hyper_parameters"]["model"]
        dataset_cfg = checkpoint["hyper_parameters"]["dataset"]

        # convert pl state dict to cuboid attention model state dict
        pl_state_dict = checkpoint["state_dict"]
        pt_state_dict = {}
        for key, val in pl_state_dict.items():
            key = key.split(".")
            if key[0] == model_cfg.model_name:
                pt_state_dict[".".join(key[1:])] = val

        model = load_model(model_cfg)
        self.model = model.load_state_dict(pt_state_dict)

        # configurable attributes
        self.data_dir = data_dir
        self.output_dir = output_dir # TODO: create if does not exists!
        self.start_time = datetime.strptime(start_time, START_TIME_FORMAT)
        assert img_format in SUPPORTED_FORMATS, "Unsupported image format!"
        self.img_format = img_format

        if left is None:
            left = dataset_cfg["preprocess"]["crop"]["left"]
        if top is None:
            top = dataset_cfg["preprocess"]["crop"]["top"]
        if width is None:
            width = dataset_cfg["preprocess"]["crop"]["width"]
        if height is None:
            height = dataset_cfg["preprocess"]["crop"]["height"]

        self.crop = dict(left=left, top=top, width=width, height=height)

        # visualization
        self.fs = fs
        self.figsize = figsize
        self.plot_stride = plot_stride

        # inference constraints
        self.time_delta = dataset_cfg["time_delta"]
        self.scale = dataset_cfg["preprocess"]["scale"]
        self.grayscale = dataset_cfg["preprocess"]["grayscale"]

    def _load_images(self):
        raw_x = []

        # open images
        curr_time = self.start_time
        time_delta = timedelta(minutes=self.time_delta)

        for i in range(self.model.in_len):
            frame_path = os.path.join(self.data_dir, f"{curr_time.strftime(IMAGE_NAME_FORMAT)}.{self.img_format}")
            if not os.path.exists(frame_path):
                print(f"Did not find input frame of time {curr_time}!")
                sys.exit()

            if self.img_format == "png":
                h, w, raw_pixels = png.Reader(file=open(frame_path, "rb")).asRGBA8()[:3]
                pixels = np.array([list(row) for row in raw_pixels], dtype="uint8").reshape((h, w, 4))
                raw_x.append(pixels)
                curr_time += time_delta

        return raw_x

    def _save_visualization(self, x, y):
        visualize = IMSVisualize(save_dir=self.output_dir,
                                 scale=self.scale,
                                 fs=self.fs,
                                 figsize=self.figsize,
                                 plot_stride=self.plot_stride)
        visualize.save_example(save_prefix=f'prediction_from_{self.start_time.strftime(IMAGE_NAME_FORMAT)}_with_ckpt_{self.ckpt_name}',
                               in_seq=x,
                               target_seq=y,
                               pred_seq_list=[y],
                               time_delta=self.time_delta)
    def inference(self):
        raw_x = self._load_images()

        preprocess = IMSPreprocess(grayscale=self.grayscale,
                                   scale=self.scale,
                                   crop=self.crop)
        x = preprocess(raw_x)
        y = self.model(x)
        self._save_visualization(x, y)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-name', default=None, type=str,
                        help="the checkpoint of the model we want to inference with."
                             "the checkpoint file is expected to be in the pretrained_checkpoints dir.")
    parser.add_argument('--data-dir', default=None, type=str,
                        help="the path where the images are at. "
                             f"the images name has to be in the following format {IMAGE_NAME_FORMAT}."
                             "the image files has to be in PNG format.")
    parser.add_argument('--start-time', default=None, type=str,
                        help=f"the time of the first frame in the input in the following format {START_TIME_FORMAT}.")  # TODO: check the validity of the start-time
    parser.add_argument('--output-dir', default="", type=str,
                        help="the path where the inference will be saved at.")
    parser.add_argument('--img-format', default="png", type=str,
                        help=f"the format of the input images.")
    parser.add_argument('--fs', default=None, type=int,
                        help=f"the font size in the visualization of the output.")
    parser.add_argument('--figsize', default=None, type=list,
                        help=f"the figure size of the visualization of the output.")
    parser.add_argument('--plot-stride', default=None, type=int,
                        help=f"the plot stride in the visualization of the output.")
    parser.add_argument('--left', default=None, type=int,
                        help=f"set where to start cropping the image from the left."
                             f"if not set, taken from checkpoint.")
    parser.add_argument('--top', default=None, type=int,
                        help=f"set where to start cropping the image from the top."
                             f"if not set, taken from checkpoint.")
    parser.add_argument('--width', default=None, type=int,
                        help=f"set the width of the cropped image."
                             f"if not set, taken from checkpoint.")
    parser.add_argument('--height', default=None, type=int,
                        help=f"set the height of the cropped image."
                             f"if not set, taken from checkpoint.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    ims_inference = CuboidIMSInference(ckpt_name=args.ckpt_name,
                                       data_dir=args.data_dir,
                                       start_time=args.start_time,
                                       output_dir=args.output_dir,
                                       img_format=args.img_format,
                                       fs=args.fs,
                                       figsize=args.figsize,
                                       plot_stride=args.plot_stride,
                                       left=args.left,
                                       top=args.top,
                                       width=args.width,
                                       height=args.height
                                       )
    ims_inference.inference()


if __name__ == "__main__":
    main()
