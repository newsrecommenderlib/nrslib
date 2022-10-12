import warnings
from torch.utils.data import DataLoader
from .download import download_and_extract_glove
from os.path import join
from .parse import parse_mind
from .dataset import (
    BaseDataset,
    NewsDataset,
    UserDataset,
    BehaviorsDataset,
)
from .datamodule_Base import MINDDataModule

from .datamodule_Base import MINDDataModule


class MINDDataModuleNAML(MINDDataModule):
    """Datamodule for the NAML model using the MIND dataset

    Code based on https://github.com/Microsoft/Recommenders

    Args:
        dataset_attributes (dict): Attributes are set based on the model
        mind_size (string): Size of the MIND Dataset (demo, small, large)
        data_dir (Optional[string]): Path of the data directory for the dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        pin_memory (bool): Requires more memory but might imporve performance
        num_clicked_news_a_user (int): Number of clicked news for each user
        num_words_title (int): Number of words in the title
        num_words_abstract (int): Number of words in the abstract
        word_freq_threshold (int): Frequency threshold of words
        entity_freq_threshold (int): Frequency threshold of entities
        entity_confidence_threshold (float): Confidence threshold of entities
        negative_sampling_ratio (int): Negative sampling ratio
        word_embedding_dim (int): Dimension of word embeddings
        entity_embedding_dim (int): Dimension of entity embeddings
        download (bool): Enable the download and extraction of the MIND dataset. When set to false, extract data must be available in data_dir.
        glove_size (int): Size of Glove embeddings to download
    """

    def __init__(
        self,
        dataset_attributes,  # config
        mind_size="small",
        data_dir=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        # config
        num_clicked_news_a_user=50,
        num_words_title=20,
        num_words_abstract=50,
        word_freq_threshold=1,
        entity_freq_threshold=2,
        entity_confidence_threshold=0.5,
        negative_sampling_ratio=2,
        word_embedding_dim=300,
        entity_embedding_dim=100,
        download=True,
        glove_size=6,
    ):

        super().__init__(
            data_dir, mind_size, batch_size, num_workers, download, pin_memory
        )
        self.pin_memory = pin_memory
        self.negative_sampling_ratio = negative_sampling_ratio
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.entity_confidence_threshold = entity_confidence_threshold
        self.entity_freq_threshold = entity_freq_threshold
        self.word_freq_threshold = word_freq_threshold
        self.entity_freq_threshold = entity_freq_threshold
        self.word_embedding_dim = word_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.num_clicked_news_a_user = num_clicked_news_a_user
        self.dataset_attributes = dataset_attributes
        self.glove_size = glove_size
        self.valid_set = {}
        self.test_set = {}
        self.train_set = {}
        self.train_path = join(self.data_dir, "train")
        self.test_path = (
            join(self.data_dir, "test") if self.mind_size == "large" else None
        )
        self.valid_path = join(self.data_dir, "valid")
        self.glove_path = join(self.data_dir, "glove")
        self.prepare_data_per_node = False
        self.data_prepared_naml = False
        self.prepare()

    def prepare(self) -> None:
        super().prepare()
        if not self.data_prepared_naml:
            if self.download:
                download_and_extract_glove(
                    dest_path=self.data_dir, glove_size=self.glove_size
                )
            parse_mind(
                self.train_path,
                self.valid_path,
                self.test_path,
                self.glove_path,
                self.glove_size,
                self.negative_sampling_ratio,
                self.num_words_title,
                self.num_words_abstract,
                self.entity_confidence_threshold,
                self.word_freq_threshold,
                self.entity_freq_threshold,
                self.word_embedding_dim,
                self.entity_embedding_dim,
            )

    def setup(self, stage):
        if not self.train_set and not self.valid_set:
            self.train_set = BaseDataset(
                join(self.train_path, "behaviors_parsed.tsv"),
                join(self.train_path, "news_parsed.tsv"),
                self.dataset_attributes,
                self.num_words_title,
                self.num_words_abstract,
                self.num_clicked_news_a_user,
            )
            self.valid_set["news"] = NewsDataset(
                join(self.valid_path, "news_parsed.tsv"), self.dataset_attributes
            )
            self.valid_set["user"] = UserDataset(
                join(self.valid_path, "behaviors.tsv"),
                join(self.train_path, "user2int.tsv"),
                self.num_clicked_news_a_user,
            )
            self.valid_set["behaviors"] = BehaviorsDataset(
                join(self.valid_path, "behaviors.tsv")
            )
        if not self.test_set and self.test_path is not None:
            self.test_set["news"] = NewsDataset(
                join(self.test_path, "news_parsed.tsv"), self.dataset_attributes
            )
            self.test_set["user"] = UserDataset(
                join(self.test_path, "behaviors.tsv"),
                join(self.train_path, "user2int.tsv"),
                self.num_clicked_news_a_user,
            )
            self.test_set["behaviors"] = BehaviorsDataset(
                join(self.test_path, "behaviors.tsv")
            )

    def news_dataloader(self, step, device=None):
        if step == "test":
            eval_set = self.test_set if self.test_path is not None else self.valid_set
        elif step == "valid":
            eval_set = self.valid_set
        else:
            raise ValueError(f"News dataloader for {step} not found!")
        if device is not None:
            eval_set["news"].to(device)
        return DataLoader(
            eval_set["news"],
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def user_dataloader(self, step):
        if step == "test":
            eval_set = self.test_set if self.test_path is not None else self.valid_set
        elif step == "valid":
            eval_set = self.valid_set
        else:
            raise ValueError(f"User dataloader for {step} not found!")
        return DataLoader(
            eval_set["user"],
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, batch_size=1):
        return DataLoader(
            self.valid_set["behaviors"],
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def test_dataloader(self, batch_size=1):
        if self.test_path is not None:
            return DataLoader(
                self.test_set["behaviors"],
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=lambda x: x,
            )
        else:
            warnings.warn(
                "Test set not available. Using validation set for testing.",
                RuntimeWarning,
            )
            return self.val_dataloader(batch_size)
