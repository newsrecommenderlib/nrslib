from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import torch


class BaseDataset(Dataset):
    """Base Dataset for training

    Args:
        behaviors_path (str): Path to behaviors file
        news_path (str): Path to news file
        dataset_attributes (list): Dataset attributes
        num_words_title (): Number of title words
        num_words_abstract (): Number of abstract words
        num_clicked_news_a_user (): Number of clicked news

    """
    def __init__(
        self,
        behaviors_path,
        news_path,
        dataset_attributes,
        num_words_title,
        num_words_abstract,
        num_clicked_news_a_user,
    ):
        super(BaseDataset, self).__init__()
        assert all(
            attribute
            in [
                "category",
                "subcategory",
                "title",
                "abstract",
                "title_entities",
                "abstract_entities",
            ]
            for attribute in dataset_attributes["news"]
        )
        assert all(
            attribute in ["user", "clicked_news_length"]
            for attribute in dataset_attributes["record"]
        )

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col="id",
            usecols=["id"] + dataset_attributes["news"],
            converters={
                attribute: literal_eval
                for attribute in set(dataset_attributes["news"])
                & set(["title", "abstract", "title_entities", "abstract_entities"])
            },
        )
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict("index")
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])
        padding_all = {
            "category": 0,
            "subcategory": 0,
            "title": [0] * num_words_title,
            "abstract": [0] * num_words_abstract,
            "title_entities": [0] * num_words_title,
            "abstract_entities": [0] * num_words_abstract,
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v for k, v in padding_all.items() if k in dataset_attributes["news"]
        }

        self.dataset_attributes = dataset_attributes
        self.num_clicked_news_a_user = num_clicked_news_a_user

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if "user" in self.dataset_attributes["record"]:
            item["user"] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [self.news2dict[x] for x in row.candidate_news.split()]
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[: self.num_clicked_news_a_user]
        ]
        if "clicked_news_length" in self.dataset_attributes["record"]:
            item["clicked_news_length"] = len(item["clicked_news"])
        repeated_times = self.num_clicked_news_a_user - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding] * repeated_times + item["clicked_news"]

        return item


class NewsDataset(Dataset):
    """News dataset for evaluation

    Args:
        news_path (str): Path to news file
        dataset_attributes (list): Dataset attributes
    """

    def __init__(self, news_path, dataset_attributes):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=["id"] + dataset_attributes["news"],
            converters={
                attribute: literal_eval
                for attribute in set(dataset_attributes["news"])
                & set(["title", "abstract", "title_entities", "abstract_entities"])
            },
        )
        self.news2dict = self.news_parsed.to_dict("index")
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2]
                    )

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item

    def to(self, device):
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = self.news2dict[key1][key2].to(device)


class UserDataset(Dataset):
    """Users dataset for evaluation. Duplicated rows will be dropped

    Args:
        behaviors_path (str): Path to behaviors file
        user2int_path (str): Path to user index file
        num_clicked_news_a_user ():
    """

    def __init__(self, behaviors_path, user2int_path, num_clicked_news_a_user):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(
            behaviors_path, header=None, usecols=[1, 3], names=["user", "clicked_news"]
        )
        self.behaviors.clicked_news.fillna(" ", inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, "user"] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, "user"] = 0
        self.num_clicked_news_a_user = num_clicked_news_a_user

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user": row.user,
            "clicked_news_string": row.clicked_news,
            "clicked_news": row.clicked_news.split()[: self.num_clicked_news_a_user],
        }
        item["clicked_news_length"] = len(item["clicked_news"])
        repeated_times = self.num_clicked_news_a_user - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ["PADDED_NEWS"] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """User behaviors dataset for evaluation. (user, time) pair as session

    Args:
        behaviors_path (str): Path to behaviors file
    """

    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(
            behaviors_path,
            header=None,
            usecols=range(5),
            names=["impression_id", "user", "time", "clicked_news", "impressions"],
        )
        self.behaviors.clicked_news.fillna(" ", inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions,
        }
        return item

class RatingsDataset(Dataset):
    """User Ratings knowledge graph dataset for dataloaders

    Args:
        numpy_data (numpy.ndarray): Ratings numpy data
        train (bool): Whether the dataset contains training data
    """
    def __init__(self, numpy_data, train: bool):
        super(RatingsDataset, self).__init__()
        self.ratings = numpy_data
        self.train = train

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings[idx]
        if self.train:
            item = {
                "user": row[0],
                "clicked_news": row[1],
                "read_status": row[2],
            }
        else:
            item = {
                "user": row[0],
                "clicked_news": row[1],
                "read_status": row[2],
                "impression_id": row[3]
            }
        return item

class KGDataset(Dataset):
    """News knowledge graph dataset for dataloaders

    Args:
        numpy_data (numpy.ndarray): Knowledge graph numpy data
    """
    def __init__(self, numpy_data):
        super(KGDataset, self).__init__()
        self.kg = numpy_data

    def __len__(self):
        return len(self.kg)

    def __getitem__(self, idx):
        row = self.kg[idx]
        item = {
            "head": row[0],
            "relation": row[1],
            "tail": row[2],
        }
        return item

class BehaviorsBERTDataset(Dataset):
    """Behaviors dataset for BERT model

        Args:
            behaviors_path (str): Path to behaviors file
        """

    def __init__(self, behaviors_path):
        super(BehaviorsBERTDataset, self).__init__()
        self.behaviors = pd.read_table(
            behaviors_path,
            # header=None,
            # usecols=range(4),
            # names=["id",	"history",	"candidate_news",	"labels"],
        )

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "id": row.id,
            "history": literal_eval(row.history),
            "candidate_news": literal_eval(row.candidate_news),
            "labels": literal_eval(row.labels)
        }
        return item

class NewsBERTDataset(Dataset):
    """News dataset for BERT model

        Args:
            news_path (str): Path to news file
        """

    def __init__(self, news_path):
        super(NewsBERTDataset, self).__init__()
        self.news = pd.read_table(
            news_path,
            # header=None,
            # usecols=range(2),
            # names=["news_id",	"text"],
        )

    def __len__(self):
        return len(self.news)

    def __getitem__(self, idx):
        row = self.news.iloc[idx]
        item = {
            "news_id": row.news_id,
            "text": row.text
        }
        return item
