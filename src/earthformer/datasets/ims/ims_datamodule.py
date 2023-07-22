from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datetime import datetime
from earthformer.datasets.ims.ims_dataset import IMSDataset

VALID_BATCH_LAYOUTS = {'NTHWC'}

class IMSLightningDataModule(LightningDataModule):
    def __init__(self,
                 start_date: datetime = None,
                 train_val_split_date: datetime = None,
                 train_test_split_date: datetime = None,
                 end_date: datetime = None,
                 num_workers: int = 1,
                 batch_size: int = 1,
                 batch_layout: str = "NTHWC",
                 **kwargs  # dataset additional parameters
                 ):
        super(IMSLightningDataModule, self).__init__()
        assert start_date <= train_val_split_date <= train_test_split_date <= end_date
        self.start_date = start_date
        self.train_val_split_date = train_val_split_date
        self.train_test_split_date = train_test_split_date
        self.end_date = end_date

        assert num_workers >= 0
        self.num_workers = num_workers
        assert batch_size >= 1
        self.batch_size = batch_size
        assert batch_layout in VALID_BATCH_LAYOUTS
        self.batch_layout = batch_layout

        self.data_set_kwargs = kwargs

    def prepare_data(self) -> None:
        # TODO: download the data
        pass

    def setup(self, stage=None) -> None:
        # read https://lightning.ai/docs/pytorch/stable/data/datamodule.html how it is supposed to look
        self.ims_train = IMSDataset(start_date=self.start_date, end_date=self.train_val_split_date,
                                    **self.data_set_kwargs)
        # TODO: delete prints
        print("train:", len(self.ims_train))
        self.ims_val = IMSDataset(start_date=self.train_val_split_date, end_date=self.train_test_split_date,
                                  shuffle=False, **self.data_set_kwargs)
        print("val:", len(self.ims_val))
        self.ims_test = IMSDataset(start_date=self.train_test_split_date, end_date=self.end_date,
                                   shuffle=False, **self.data_set_kwargs)
        print("test:", len(self.ims_test))
        # TODO: test and predict are the same
        self.ims_predict = IMSDataset(start_date=self.train_test_split_date, end_date=self.end_date,
                                      **self.data_set_kwargs)

    def train_dataloader(self):
        return DataLoader(self.ims_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ims_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ims_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ims_predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_hwc(self):
        return self.ims_train.img_shape
