import argparse
import os
import sys
import png
import numpy as np
from datetime import datetime, timedelta

import torch

from earthformer.utils.ims.load_ims import load_model
from earthformer.config import cfg
from earthformer.datasets.ims.ims_dataset import IMSPreprocess
from earthformer.visualization.ims.ims_visualize import IMSVisualize

START_TIME_FORMAT = "%Y%m%d%H%M"
IMAGE_NAME_FORMAT = "%Y%m%d%H%M"

pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir


class CuboidIMSInference:
    def __init__(self,
                 ckpt: str,
                 data_dir: str,
                 start_time: str,
                 fs: int,
                 figsize: tuple,
                 plot_stride: int,
                 cmap: str,
                 left: int,
                 top: int,
                 width: int,
                 height: int,
                 output_dir: str = './'):
        """
        ckpt: The path of the checkpoint we want to load.
        data_dir: The path of directory containing the images.
        start_time: The time of the first frame in the sequence.
        output_dir: The output directory, the output is a summary file of the prediction.
        fs: The font size in the summary.
        figsize: The size of the images in the summary.
        plot_stride: The "jumps" between frames in the summary.
        cmap: the cmap that is used for the images in the summary.
        left, top, width, height: Crop input images parameters.
        """
        self.ckpt = ckpt
        self.ckpt_name = os.path.basename(ckpt)

        # load model
        if os.path.exists(ckpt):
            checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
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

        self.model = load_model(model_cfg)
        self.model.load_state_dict(pt_state_dict)
        self.model.eval()

        # configurable attributes
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # creates output_dir if does not exist
        self.start_time = datetime.strptime(start_time, START_TIME_FORMAT)

        # take checkpoint defaults if None
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
        self.cmap = cmap

        # inference constraints
        self.in_len = model_cfg["in_len"]
        self.time_delta = dataset_cfg["time_delta"]
        self.scale = dataset_cfg["preprocess"]["scale"]
        self.grayscale = dataset_cfg["preprocess"]["grayscale"]

    def _load_images(self):
        raw_x = []

        # open images
        curr_time = self.start_time
        time_delta = timedelta(minutes=self.time_delta)

        for i in range(self.in_len):
            frame_path = os.path.join(self.data_dir, f"{curr_time.strftime(IMAGE_NAME_FORMAT)}.png")
            if not os.path.exists(frame_path):
                print(f"Did not find input frame of time {curr_time}!")
                sys.exit()

            h, w, raw_pixels = png.Reader(file=open(frame_path, "rb")).asRGBA8()[:3]
            pixels = np.array([list(row) for row in raw_pixels], dtype="float32").reshape((h, w, 4))
            raw_x.append(pixels)
            curr_time += time_delta

        return raw_x

    def _save_visualization(self, x, y):
        visualize = IMSVisualize(save_dir=self.output_dir,
                                 fs=self.fs,
                                 figsize=self.figsize,
                                 plot_stride=self.plot_stride,
                                 cmap=self.cmap)
        visualize.save_example(
            save_prefix=f'prediction_from_{self.start_time.strftime(IMAGE_NAME_FORMAT)}_with_ckpt_{self.ckpt_name.split(".")[0]}',
            in_seq=x,
            pred_seq_list=[y],
            label_list=[self.ckpt_name],
            time_delta=self.time_delta)

    def inference(self):
        raw_x = self._load_images()
        x = self._preprocess(raw_x)
        y = self.model(x)
        # the batch size is 1, detach from model
        # clip predicted values to be between 0 and 1
        y = torch.clip(y, min=0.0, max=1.0)
        # save output
        self._save_visualization(x[0].detach().numpy(), y[0].detach().numpy())

    def _preprocess(self, raw_x):
        # raw_x is a sequence of shape THWC
        preprocess = IMSPreprocess(grayscale=self.grayscale,
                                   scale=self.scale,
                                   crop=self.crop)
        x = preprocess(raw_x)
        # x is a tensor of shape THWC, needs to be converted to batch of shape NTHWC
        return torch.stack([x])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, type=str, required=True,
                        help="the checkpoint of the model we want to inference with.")
    parser.add_argument('--data-dir', default=None, type=str, required=True,
                        help="the path where the images are at. "
                             f"the images name has to be in the following format {IMAGE_NAME_FORMAT}."
                             "the image files has to be in PNG format.")
    parser.add_argument('--start-time', default=None, type=str, required=True,
                        help=f"the time of the first frame in the input in the following format {START_TIME_FORMAT}.")  #TODO: check the validity of the start-time
    parser.add_argument('--output-dir', default="./", type=str,
                        help="the path where the inference will be saved at.")
    parser.add_argument('--fs', default=None, type=int,
                        help=f"the font size in the visualization of the output.")
    parser.add_argument('--figsize', default=None, type=list,
                        help=f"the figure size (height and width, separated with a space) of the visualization of the "
                             f"output.")
    parser.add_argument('--plot-stride', default=None, type=int,
                        help=f"the plot stride in the visualization of the output.")
    parser.add_argument('--cmap', default=None, type=str,
                        help=f"the cmap of the images in the visualization.")
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
    ims_inference = CuboidIMSInference(**vars(parser.parse_args()))
    ims_inference.inference()


if __name__ == "__main__":
    main()
