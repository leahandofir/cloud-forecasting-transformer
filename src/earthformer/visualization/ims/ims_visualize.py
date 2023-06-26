import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
# from .ims_cmap import get_cmap

# TODO: try to use the cmap

class IMSVisualize:
    def __init__(self,
                 save_dir: str = "./",
                 scale: bool = True,
                 fs: int = None,
                 figsize: tuple = None,
                 plot_stride: int = None,
                 ):

        self.save_dir = save_dir
        self.scale = 255 if scale else 1

        if fs is None:
            fs = 10
        self.fs = fs

        if figsize is None:
            figsize = (24, 8)
        self.figsize = tuple(figsize)

        if plot_stride is None:
            plot_stride = 2
        self.plot_stride = plot_stride

    def _plot_seq(self, ax, row, label, seq, seq_len, max_len):
        ax[row][0].set_ylabel(label, fontsize=self.fs)
        for i in range(0, max_len, self.plot_stride):
            if i < seq_len:
                xt = seq[i, :, :, :] * (self.scale)
                ax[row][i // self.plot_stride].imshow(xt)
            else:
                ax[row][i // self.plot_stride].axis('off')

    def _visualize_result(self,
                          in_seq: np.array,
                          pred_seq_list: List[np.array],
                          label_list: List[str],
                          time_delta: int = 5,
                          target_seq: np.array = None,
                          ):
        """
        in_seq, target_seq are from shape THWC.
        pred_seq_list is a list of sequences from shape THWC.
        All sequences have to be at the same dimensions as the target sequence.
        It is not mandatory to plot the target sequence, if target_seq is None the target is not plotted.
        """
        # determine amount of subplots
        in_len = in_seq.shape[0]
        out_len = pred_seq_list[0].shape[0]
        max_len = max(in_len, out_len)
        pred_idx = 2 if target_seq is not None else 1
        nrows = pred_idx + len(pred_seq_list)
        ncols = (max_len - 1) // self.plot_stride + 1

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

        # plot all sequences
        self._plot_seq(ax, 0, "Inputs", in_seq, in_len, max_len)
        if target_seq is not None:
            self._plot_seq(ax, 1, "Target", target_seq, out_len, max_len)
        for k in range(len(pred_seq_list)):
            self._plot_seq(ax, k + pred_idx, label_list[k] + "\nPrediction", pred_seq_list[k], out_len, max_len)

        # write minutes labels
        for i in range(0, out_len, self.plot_stride):
            ax[-1][i // self.plot_stride].set_title(f'{int(time_delta * (i + self.plot_stride))} Minutes', y=-0.25)

        # remove ticks
        for i in range(nrows):
            for j in range(ncols):
                ax[i][j].xaxis.set_ticks([])
                ax[i][j].yaxis.set_ticks([])

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        return fig, ax

    def save_example(self,
                     save_prefix,
                     in_seq: np.array,
                     pred_seq_list: List[np.array],
                     label_list: List[str] = ["cuboid_ims"],
                     time_delta: int = 5,
                     target_seq: np.array = None,
                     ):
        fig_path = os.path.join(self.save_dir, f'{save_prefix}.png')

        fig, ax = self._visualize_result(
            in_seq=in_seq,
            target_seq=target_seq,
            pred_seq_list=pred_seq_list,
            label_list=label_list,
            time_delta=time_delta)

        plt.savefig(fig_path)
        plt.close(fig)
