import warnings
from os.path import join
from .parse import parse_mind_bert
from .dataset import (
    BehaviorsBERTDataset,
    NewsBERTDataset
)

from torch.utils.data import DataLoader
from .datamodule_Base import MINDDataModule


class MINDDataModuleBERT(MINDDataModule):
    def __init__(
        self,
        mind_size="demo",
        data_dir=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        # Config
        download=True,
        column="title",
        bert_model=None,
        tokenizer=None
    ):
        super().__init__(
            data_dir,
            mind_size,
            batch_size,
            num_workers,
            download,
            pin_memory
        )
        self.pin_memory = pin_memory
        self.valid_set = {}
        self.test_set = {}
        self.train_set = {}
        self.train_path = join(self.data_dir, "train")
        self.test_path = (
            join(self.data_dir, "test") if self.mind_size == "large" else None
        )
        self.valid_path = join(self.data_dir, "valid")
        self.prepare_data_per_node = False
        assert column in ['title', 'abstract'], f" column: {column} should be either title or abstract"
        self.column = column

    def prepare_data(self) -> None:
        super().prepare_data()
        self.train_news, self.val_news, self.test_news=parse_mind_bert(self.train_path,
            self.valid_path,
            self.test_path,
            self.column)


    def setup(self, stage=None):
        if not self.train_set and not self.valid_set:
            self.train_set["behaviors"] = BehaviorsBERTDataset(
                join(self.train_path, "behaviors_BERT_parsed.tsv")
            )
            self.train_set["news"] = self.train_news
            self.valid_set["behaviors"]= BehaviorsBERTDataset(
                join(self.valid_path, "behaviors_BERT_parsed.tsv")
            )
            self.valid_set["news"] = self.val_news
        if not self.test_set and self.test_path is not None:
            self.test_set["behaviors"] = BehaviorsBERTDataset(
                join(self.test_path, "behaviors_BERT_parsed.tsv")
            )
            self.test_set["news"] = self.test_news

    def news_dataframe(self, step, device=None):
        if step == "test":
            eval_set = self.test_set if self.test_path is not None else self.valid_set
        elif step == "valid":
            eval_set = self.valid_set
        elif step == "train":
            eval_set = self.train_set
        else:
            raise ValueError(f"News dataloader for {step} not found!")
        # if device is not None:
        #     eval_set["news"].to(device)
        return eval_set["news"]

    def train_dataloader(self, batch_size=1):
        return DataLoader(
            self.train_set["behaviors"],
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
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
