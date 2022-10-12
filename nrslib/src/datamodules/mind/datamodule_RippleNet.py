from .preprocessing import (
    create_knowledge_graph_file,
    create_rating_file,
    prepare_numpy_data,
)
from .download import download_and_extract_wikidata_kg
from .dataset import RatingsDataset
from .datamodule_Base import MINDDataModule
from os.path import join
from torch.utils.data import DataLoader


class MINDDataModuleRippleNet(MINDDataModule):
    """Datamodule for the RippleNet model using the MIND dataset

        Args:
            use_categories (bool): Whether the data preprocessing includes news categories
            use_subcategories (bool): Whether the data preprocessing includes news subcategories
            use_title_entities (bool): Whether the data preprocessing includes news title entities
            use_abstract_entities (bool): Whether the data preprocessing includes news abstract entities
            use_title_tokens (bool): Whether the data preprocessing includes news title tokens
            use_wikidata (bool): Whether the data preprocessing includes additional news entity wikidata knowledge graph
            data_dir (str): Data directory
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            pin_memory (bool): Whether to use pin memory
            download (bool): Whether the mind dataset should be downloaded
            mind_size (str): Dataset size
        """

    def __init__(
        self,
        use_categories,
        use_subcategories,
        use_title_entities,
        use_abstract_entities,
        use_title_tokens,
        use_wikidata,
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
        self.use_categories = use_categories
        self.use_subcategories = use_subcategories
        self.use_title_entities = use_title_entities
        self.use_abstract_entities = use_abstract_entities
        self.use_title_tokens = use_title_tokens
        self.use_wikidata = use_wikidata

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

    def prepare_data(self) -> None:
        pass

    def prepare(self):
        """Prepare data for model usage, prior to model instantiation
        Download wikidata knowledge graph, create knowledge graph and ratings files for training, validation, testing.
        """
        super().prepare_data()
        if self.use_wikidata:
            download_and_extract_wikidata_kg(
                self.data_dir, clean_zip_file=False
            )

        path_to_kg, path_to_item_ids = create_knowledge_graph_file(
            "rippleNet",
            [self.train_path, self.valid_path, self.test_path],
            self.use_categories,
            self.use_subcategories,
            self.use_title_entities,
            self.use_abstract_entities,
            self.use_title_tokens,
            self.use_wikidata,
        )

        if not self.train_set and not self.valid_set and not self.test_set:
            create_rating_file(
                [
                    self.train_path,
                    self.valid_path,
                    self.test_path,
                ],
                path_to_item_ids,
                "rippleNet",
            )

    def setup(self, stage=None):
        """Create ratings datasets for dataloaders.
        """
        if not self.train_set and not self.valid_set:
            self.train_set = RatingsDataset(
                prepare_numpy_data(self.train_path, "rippleNet"), train=True
            )
            self.valid_set = RatingsDataset(
                prepare_numpy_data(self.valid_path, "rippleNet"), train=False
            )

        if not self.test_set and self.test_path is not None:
            self.test_set = RatingsDataset(
                prepare_numpy_data(self.test_path, "rippleNet"), train=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
