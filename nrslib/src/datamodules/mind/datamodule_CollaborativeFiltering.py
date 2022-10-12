from .preprocessing import (
    create_rating_file_collaborative,
    prepare_numpy_data,
)
from .dataset import RatingsDataset
from .datamodule_Base import MINDDataModule
from os.path import join
from torch.utils.data import DataLoader


class MINDDataModuleCollaborativeFiltering(MINDDataModule):
    """Datamodule for the Collaborative Filtering model using the MIND dataset

    Args:
        data_dir (str): Data directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        pin_memory (bool): Whether to use pin memory
        download (bool): Whether the mind dataset should be downloaded
        mind_size (str): Dataset size
     """

    def __init__(
            self,
            data_dir,
            batch_size,
            num_workers,
            pin_memory,
            download,
            mind_size,
    ):
        super().__init__(
            data_dir,
            mind_size,
            batch_size,
            num_workers,
            download,
            pin_memory,
        )

        self.valid_set = None
        self.test_set = None
        self.train_set = None

        self.train_path = join(self.data_dir, "train")
        self.valid_path = join(self.data_dir, "valid")
        self.test_path = (
            join(self.data_dir, "test") if self.mind_size == "large" else None
        )

        self.prepare_data_per_node = False
        self.prepare_data()
        self.prepare()

    def prepare(self):
        """Prepare data for model usage, prior to model instantiation.
        Create ratings files for training, validation, testing.
        """
        super().prepare_data()
        create_rating_file_collaborative([self.train_path, self.valid_path, self.test_path], r"collaborative")

    def prepare_data(self):  #
        pass

    def setup(self, stage=None):
        """Create ratings datasets, knowledge graph dataset for dataloaders.
        """
        if not self.train_set and not self.valid_set:
            self.train_set = RatingsDataset(
                prepare_numpy_data(self.train_path, "collaborative"), train=True
            )
            self.valid_set = RatingsDataset(
                prepare_numpy_data(self.valid_path, "collaborative"), train=False
            )

        if not self.test_set and self.test_path is not None:
            self.test_set = RatingsDataset(
                prepare_numpy_data(self.test_path, "collaborative"), train=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
