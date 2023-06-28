from earthformer.datasets.ims.ims_dataset import IMSDataset
import matplotlib.pyplot as plt
import cv2
from omegaconf import OmegaConf
from earthformer.utils.ims.load import load_dataset_params

class IMSHistogram:
    def __init__(self, cfg_file_path):
        cfg = OmegaConf.load(open(cfg_file_path, "r"))
        self.dataset = IMSDataset(**load_dataset_params(cfg))

    def plot_hist(self, idx_sample):
        # https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
        start_time, seq = self.dataset[idx_sample]
        # plot histogram, data is of shape NTHWC
        hist = cv2.calcHist(images=seq.numpy(),
                            channels=[4],
                            mask=None,
                            histSize=[256],
                            ranges=[0,256])
        plt.plot(hist)
        plt.savefig('hist_ir.png')
