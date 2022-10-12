import warnings
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .download import download_and_extract_mind
import os
from os.path import join


class MINDDataModule(pl.LightningDataModule):
    """Base Datamodule for the MIND dataset

        Args:
            data_dir (str): Data directory
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            pin_memory (bool): Whether to use pin memory
            download (bool): Whether the mind dataset should be downloaded
            mind_size (str): Which dataset size should be used
            train_val_test_split (list): Whether to use automatic train-validation-test data splits

    """
    def __init__(
        self,
        data_dir,
        mind_size,
        batch_size,
        num_workers,
        download,
        pin_memory,
    ):
        super().__init__()
        self.mind_size = mind_size
        self.data_dir = (
            os.path.normpath(data_dir) if data_dir is not None else os.getcwd()
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.pin_memory = pin_memory
        self.valid_set = None
        self.train_set = None
        self.test_set = None
        self.train_path = join(self.data_dir, "train")
        self.test_path = (
            join(self.data_dir, "test") if self.mind_size == "large" else None
        )
        self.valid_path = join(self.data_dir, "valid")
        self.prepare_data_per_node = False
        self.data_prepared = False

    def prepare(self) -> None:
        if self.download and not self.data_prepared: 
            download_and_extract_mind(size=self.mind_size, dest_path=self.data_dir)
        self.data_prepared = True

    def setup(self, stage=None):
        warnings.warn(
            "Setup function is not implemented for current class", RuntimeWarning
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_path is not None:
            return DataLoader(
                self.test_set,
                self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            warnings.warn(
                "Test set not available. Using validation set for testing.",
                RuntimeWarning,
            )
            return self.val_dataloader()
