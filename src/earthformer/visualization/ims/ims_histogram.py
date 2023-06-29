from earthformer.datasets.ims.ims_dataset import IMSDataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from omegaconf import OmegaConf
from earthformer.utils.ims.load import load_dataset_params

class IMSHistogram:
    def __init__(self, cfg_file_path):
        cfg = OmegaConf.load(open(cfg_file_path, "r"))
        self.dataset = IMSDataset(**load_dataset_params(cfg))

    def plot_hist(self, sample_count=5):
        # https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html

        sample_indices = np.random.choice(range(len(self.dataset)), size=sample_count, replace=False)
        seq_samples = np.stack([self.dataset[i][1] for i in sample_indices])
        
        # plot histogram, data is of shape NTHWC
        hist = cv2.calcHist(images=seq_samples,
                            channels=[4],
                            mask=None,
                            histSize=[256],
                            ranges=[0,256])
        plt.plot(hist)
        plt.savefig('hist_vis.png')


# ------------------------------------------------------------------------------


ims_hist_obj = IMSHistogram(cfg_file_path="src\\earthformer\\visualization\\ims\\cfg_ims_dataset.yaml")

ims_hist_obj.plot_hist()

