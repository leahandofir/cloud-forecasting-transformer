import matplotlib.pyplot as plt
import cv2, os, torch
import numpy as np
from omegaconf import OmegaConf

from earthformer.utils.ims.load import load_dataset_params
from earthformer.datasets.ims.ims_dataset import IMSDataset


class IMSHistogram:
    def __init__(self, cfg_file_path):
        self.cfg = OmegaConf.load(open(cfg_file_path, "r"))
        self.dataset = IMSDataset(**load_dataset_params(self.cfg))

    def plot_hist(self, sample_count=5, output_path="./", idx_sample=None):
        # https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
        if idx_sample is None:
            sample_indices = np.random.choice(range(len(self.dataset)), size=sample_count, replace=False)
            # reshape to (#samples * T)xHxWxC
            seq_samples = torch.flatten(torch.stack([self.dataset[i][1] for i in sample_indices]), start_dim=0, end_dim=1)
        else:
            seq_samples = self.dataset[idx_sample][1]

        seq_samples = [img for img in seq_samples.numpy()]
        
        # plot histogram, data is of shape (#samples * T)xHxWxC
        hist = cv2.calcHist(images=seq_samples,
                            channels=[0],
                            mask=None,
                            histSize=[256],
                            ranges=[0,256])
        plt.plot(hist)
        plt.savefig(os.path.join(output_path, f"hist_samples={sample_count}_{self.cfg.img_type}.png"))
