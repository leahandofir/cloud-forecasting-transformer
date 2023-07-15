from typing import Sequence
import torch
from torchmetrics import Metric


class IMSSkillScore(Metric):
    r"""
    We calculated the MAE and mCSI according to SEVIR challenge:
    https://www.cawcr.gov.au/projects/verification/#Methods_for_spatial_forecasts
    https://github.com/MIT-AI-Accelerator/sevir_challenges/blob/dev/radar_nowcasting/RadarNowcastBenchmarks.ipynb
    We assume the layout of the predicated sequence and the target sequence is NTHWC.
    We added an option to consider the MSE.
    """

    def __init__(self,
                 scale: bool = True,
                 threshold_list: Sequence[int] = (130, 160),
                 threshold_weights: Sequence[int] = None,
                 metrics_list: Sequence[str] = ("csi", "mae", "mse"),
                 ):
        super().__init__()
        self.scale = scale
        self.threshold_count = len(threshold_list)
        if threshold_weights is None:
            threshold_weights = torch.tensor([1.0 / self.threshold_count] * self.threshold_count)

        assert self.threshold_count == len(threshold_weights)
        self.threshold_list = threshold_list
        assert sum(threshold_weights) == 1.0
        self.threshold_weights = threshold_weights
        self.metrics_list = metrics_list
        self.add_state("hits",
                       default=torch.zeros(self.threshold_count),
                       dist_reduce_fx="sum")
        self.add_state("misses",
                       default=torch.zeros(self.threshold_count),
                       dist_reduce_fx="sum")
        self.add_state("fas",
                       default=torch.zeros(self.threshold_count),
                       dist_reduce_fx="sum")
        self.add_state("total_numel",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("sum_abs_error",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("sum_squared_error",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")

    def csi(self):
        # calculate CSI for each threshold and calculate weighted average
        csi_per_threshold = torch.zeros(self.threshold_count)
        for i in range(self.threshold_count):
            csi_per_threshold[i] = self.hits[i] / (self.hits[i] + self.misses[i] + self.fas[i])
        return 1 - torch.sum(csi_per_threshold * self.threshold_weights) # because it is going to be minimized

    def mae(self):
        return self.sum_abs_error / self.total_numel

    def mse(self):
        return self.sum_squared_error / self.total_numel

    def _calc(self, prediction, target, threshold):
        """
        Calculate hits, misses, and false alarms according to the given threshold, pixel by pixel.
        """
        with torch.no_grad():
            t = (target >= threshold).float()
            p = (prediction >= threshold).float()
            hits = torch.sum(t * p).int()
            misses = torch.sum(t * (1 - p)).int()
            fas = torch.sum((1 - t) * p).int()
        return hits, misses, fas

    def update(self, prediction, target):
        # rescale
        prediction = prediction * 255.0 if self.scale else prediction * 1.0
        target = target * 255.0 if self.scale else target * 1.0

        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self._calc(prediction, target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas

        self.total_numel += target.numel()
        self.sum_abs_error += torch.sum(torch.abs(prediction - target))
        self.sum_squared_error += torch.sum((prediction - target)**2)

    def compute(self):
        """
        Returns a torch.tensor with a score to each metric.
        """
        metrics_dict = {
            'csi': self.csi,
            'mae': self.mae,
            'mse': self.mse}

        scores = torch.zeros(len(self.metrics_list))
        for i, metric in enumerate(self.metrics_list):
            scores[i] = metrics_dict[metric]()
        return scores
