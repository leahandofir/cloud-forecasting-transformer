# TODO: maybe it is more efficient to read subsequent samples.
# TODO: verify that we took every important thing from SEVIR code

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np
import datetime, h5py, os
from typing import List, Union, Dict, Sequence
from src.earthformer.config import cfg

# IMS dataset constants
IMS_IMG_TYPES = {"MIDDLE_EAST_VIS", "MIDDLE_EAST_DAY_CLOUDS", "MIDDLE_EAST_COLORED", "MIDDLE_EAST_IR"}
VALID_LAYOUTS = {'THWC'}
VALID_CHANNELS = (1, 3, 4)

# IMS dataset directory
# the structure of the data directory must be as follows: img_type -> year -> h5 file
IMS_ROOT_DIR = os.path.join(cfg.datasets_dir, "ims")
IMS_CATALOG = os.path.join(IMS_ROOT_DIR, "CATALOG.csv")
IMS_DATA_DIR = os.path.join(IMS_ROOT_DIR, "data")


class IMSDataset(Dataset):
    def __init__(self,
                 img_type: str = 'MIDDLE_EAST_VIS',
                 seq_len: int = 49,
                 raw_seq_len: int = 169,
                 stride: int = 12,
                 layout: str = 'THWC',
                 raw_img_shape: Union[tuple, list] = (600, 600, 1),
                 ims_catalog: Union[str, pd.DataFrame] = None,
                 ims_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 time_delta: int = 5,
                 raw_time_delta: int = 5,
                 shuffle: bool = False,  # gives randomness resolution to the event level (between events)
                 shuffle_seed: int = 1,
                 grayscale: bool = False,
                 left: int = 0,
                 top: int = 0,
                 width: int = None,
                 height: int = None,
                 scale: bool = True):

        super(IMSDataset, self).__init__()

        # files and directories parameters
        if ims_catalog is None:
            ims_catalog = IMS_CATALOG
        if ims_data_dir is None:
            ims_data_dir = IMS_DATA_DIR
        if isinstance(ims_catalog, str):
            self.catalog = pd.read_csv(ims_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = ims_catalog
        self.ims_data_dir = ims_data_dir

        # data parameters
        # TODO: consider including time filter.
        self.raw_seq_len = raw_seq_len
        assert img_type in IMS_IMG_TYPES, 'Invalid image type!'
        self.img_type = img_type
        self.start_date = start_date
        self.end_date = end_date
        assert time_delta % raw_time_delta == 0
        self.time_delta = time_delta
        self.raw_time_delta = raw_time_delta
        if layout not in VALID_LAYOUTS:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {VALID_LAYOUTS}.')
        self.layout = layout

        # samples parameters
        assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        self.seq_len = seq_len
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)

        assert len(raw_img_shape) == 3  # we assume the images are in HWC dimensions
        self.raw_img_shape = raw_img_shape
        max_width = raw_img_shape[0]
        assert 0 <= left
        if width is not None:
            assert 0 <= left + width <= max_width
        else:
            width = max_width

        max_height = raw_img_shape[1]
        assert 0 <= top
        if height is not None:
            assert 0 <= top + height <= max_height
        else:
            height = max_height

        channels = raw_img_shape[2]
        assert channels in VALID_CHANNELS
        self.channels = channels

        self.img_shape = (width, height, 1 if grayscale else self.channels)
        self.shift = (left, top)

        self.preprocess = IMSPreprocess(grayscale=grayscale, crop=dict(left=left, top=top, width=width, height=height),
                                        scale=scale)

        # setup
        self._events = None
        self._hdf_files = {}

        self._load_events()
        self._open_files()

    def _load_events(self):
        self._events = self.catalog

        # convert time_utc column to datetime
        def correct_datetime_format(s):  # string date, with or without hour
            date_parts = s.split(" ")
            if len(date_parts) == 1:  # date with no hour
                return f'{s} 00:00:00'
            return s

        self._events['time_utc'] = self._events['time_utc'].apply(lambda s: correct_datetime_format(s))
        self._events['time_utc'] = pd.to_datetime(self._events['time_utc'])

        # filter catalog file to contain only the relevant dates and img_type
        if self.start_date is not None:
            self._events = self._events[self._events.time_utc >= self.start_date]
        if self.end_date is not None:
            self._events = self._events[self._events.time_utc <= self.end_date]

        self._events = self._events[self._events.img_type == self.img_type]

        if self.shuffle:
            self._events = self._events.sample(frac=1, random_state=self.shuffle_seed)

    def _open_files(self):
        # open file descriptors for all the relevant h5 files containing the data
        file_names = self._events['file_name'].unique()
        for f in file_names:
            events = self._events[self._events['file_name'] == f]
            img_type = events.iloc[0]['img_type']
            year = str(events.iloc[0]['time_utc'].year)
            self._hdf_files[f] = h5py.File(os.path.join(self.ims_data_dir, img_type, year, f), 'r')

    def _idx_sample(self, index):
        event_idx = index // self.num_seq_per_event
        seq_idx = index % self.num_seq_per_event
        event = self._events.iloc[event_idx]
        raw_seq = self._hdf_files[event['file_name']][self.img_type][event['file_index']]
        raw_seq_start_time = event['time_utc']

        step_idx = self.time_delta // self.raw_time_delta
        start_idx = seq_idx * self.stride
        stop_idx = start_idx + self.real_sequence_len
        slice_sample = slice(start_idx, stop_idx, step_idx)

        seq_start_time = raw_seq_start_time + datetime.timedelta(minutes=self.raw_time_delta) * start_idx

        seq = raw_seq[slice_sample, :, :, :]  # TODO: allow layout different then THWC
        return seq_start_time, seq

    def close(self):
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.real_sequence_len) // self.stride

    @property
    def total_num_event(self):
        return int(self._events.shape[0])

    @property
    def total_num_seq(self):
        return int(self.num_seq_per_event * self.total_num_event)

    @property
    def real_sequence_len(self):
        return (self.seq_len - 1) * (self.time_delta // self.raw_time_delta) + 1  # counting the frames we skip

    def __len__(self):
        return self.total_num_seq

    def __getitem__(self, index):
        start_time, sample = self._idx_sample(index)
        if self.preprocess:
            sample = self.preprocess(sample)

        return datetime.datetime.timestamp(start_time), sample


class IMSPreprocess:
    def __init__(self, grayscale=False, crop={}, scale=True, data_type=torch.float32):
        # build the transformation function according to the parameters
        # convert (H x W x C) to a Tensor (C x H x W)
        relevant_transforms = [transforms.ToTensor()]

        # scaling to [0.0, 1.0] (or not)
        if scale:
            relevant_transforms.append(transforms.Lambda(lambda t: t / 255))

        # convert to grayscale (1 x H x W) if necessary
        if grayscale:
            relevant_transforms.append(
                transforms.Lambda(lambda x: transforms.Grayscale(x[:3, :, :]) if x.shape[0] > 1 else x))

        # crop image if necessary
        if len(crop.keys()) > 0:
            relevant_transforms.append(transforms.Lambda(
                lambda t: F.crop(t, crop['top'], crop['left'], crop['height'], crop['width'])))

        # convert Tensor (C x H x W) to a Tensor (H x W x C)
        relevant_transforms.append(transforms.Lambda(
            lambda t: torch.moveaxis(t, -3, -1)))

        # convert data type
        relevant_transforms.append(transforms.Lambda(
            lambda t: t.to(data_type)))

        # save the final transformation function
        self.preprocess_frame = transforms.Compose(relevant_transforms)

    def preprocess_seq(self, seq):
        return torch.stack([self.preprocess_frame(frame) for frame in seq])

    def __call__(self, x):  # x is a sequence with dimensions THWC
        return self.preprocess_seq(x)
