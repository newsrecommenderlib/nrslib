
import warnings
import os
from os.path import join

import numpy as np
from ...datamodules.mind.download import download_and_extract_mind
import pandas as pd
import string
import torch
from transformers import BertTokenizer, BertModel, logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import datetime
import inspect

logging.set_verbosity_error()
import logging


class BERT:
    """Content Based Recommendation based on BERT embeddings

    Using BERT embeddings to calculate the cosine similarity between the user history and the candidate news

    Args:
         dataset(String): name of the dataset used. ex: "mind"
         news_df(pandas.DataFrame): represents the news. Must have a ['news_id', 'text'] columns, When using dataset other than MIND
         behaviors_df(pandas.DataFrame): represents the behaviors of the users. Must have a ['id', 'history', 'candidate_news', 'labels'] columns,When using dataset other than MIND
         mind_size(String):the size of the data set to get the results from. Possible values: ['small', 'demo','large'],When using the MIND dataset
         part(String): the part which the results model is using Possible values: ['valid', 'test','train'],When using the MIND dataset
         column(String):Possible values: ['title', 'abstract'],When using the MIND dataset
         download(Boolean): for downloading the MIND dataset,When using the MIND dataset
         bert_model: for using another pre-trained model. Default="bert-base-uncased",
         tokenizer: for using another tokenizer. Default="bert-base-uncased",
         metrics(set): the set of metrics from torchmetrics for evaluation
    """
    def __init__(self,
                 # using another dataset
                 news_df=None,
                 behaviors_df=None,
                 # using MIND dataset
                 dataset="mind",
                 mind_size="demo",
                 part="valid",
                 column="title",
                 download=True,
                 # config
                 bert_model=None,
                 tokenizer=None,
                 metrics={}):
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        valid_metrics = []
        self.metrics_params = []
        for name, param in metrics.items():
            d = {}
            target = param.pop("_target_")
            exec(f"metric = {target}(**{str(param)})", globals(), d)
            args = inspect.signature(d["metric"].update).parameters.keys()
            valid_metrics.append(d["metric"])
            self.metrics_params.append((name, "indexes" in args))
        self.valid_metrics = torch.nn.ModuleList(valid_metrics)
        self.writer = SummaryWriter(
            log_dir=
            f"tensorboard/baseline_BERT/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        if dataset != "mind":
            assert 'news_id' in news_df, f" news_df should have a news_id column "
            assert 'text' in news_df, f" news_df should have a text column "
            self.news_df = news_df
            assert 'id' in behaviors_df, f" behaviors_df should have an id column "
            assert 'history' in behaviors_df, f" behaviors_df should have a history column "
            assert 'candidate_news' in behaviors_df, f" behaviors_df should have an candidate_news column "
            assert 'labels' in behaviors_df, f" behaviors_df should have a labels column "
            self.behaviors_df = behaviors_df
            pass
        else:
            self.download = download
            assert mind_size in ['small', 'demo',
                                 'large'], f" mind_size: {mind_size} should be either small, demo or large "
            self.mind_size = mind_size
            assert part in ['valid', 'train', 'test'], f" part: {part} should be either valid, train or test"
            if part == 'test' and mind_size != "large":
                raise ValueError(
                    f" part: {part} is not supported with mind_size: {mind_size},change one of the variables ")
            self.part = part
            assert column in ['title', 'abstract'], f" column: {column} should be either title or abstract"
            self.column = column
            self.data_dir = join(os.getcwd(), "data/MIND")
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        if bert_model is None:
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert_model = bert_model

    def prepare_data(self) -> None:
        """
         This function is for prepare the data for the mind dataset only.
         It does:
            - Download MIND
            - parse news.tsv
            - Parse behaviors.tsv
        """
        if self.dataset == "mind":
            if self.download:
                download_and_extract_mind(size=self.mind_size, dest_path=self.data_dir)
            part_dir = join(self.data_dir, self.part)
            self.news_df = self.parse_news(
                join(part_dir, "news.tsv"),
                join(part_dir, "news_BERT_parsed.tsv"))
            self.behaviors_df = self.parse_behaviors(
                join(part_dir, "behaviors.tsv"),
                join(part_dir, "behaviors_BERT_parsed.tsv"))
        # print(f"Generating the BERT embedding")
        # ids2embedding={}
        # # embeddings = []
        # for id,text in tqdm(zip(self.news_df.news_id,self.news_df.text), desc="Calculating text embeddings"):
        #     ids2embedding[id]=self.get_bert_embeddings(text, self.tokenizer, self.bert_model)
        # # self.news_df['text_embedding'] = embeddings
        # print(f"Done: Generating the BERT embedding")

    def parse_news(self, source, target):
        """
        Parse news for using BERT baseline
        Generate BERT embedding for the text in news df

        Args:
            source: source news file
            target: target news file

        Returns:
            DataFrame: news_parsed(news_id, text)

        """
        print(f"Parse {source}")
        news = pd.read_table(
            source,
            header=None,
            usecols=[0, 1, 2, 3, 4, 6, 7],
            # quoting=csv.QUOTE_NONE,
            names=[
                "id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "title_entities",
                "abstract_entities",
            ],
        )
        news.fillna(" ", inplace=True)
        parsed_news = pd.DataFrame()
        parsed_news["news_id"] = news.id
        parsed_news["text"] = news[self.column]

        def clean(text):
            # Remove punctuation from the text
            remove_punctuation_dict = dict((ord(mark), None) for mark in string.punctuation)
            return text.translate(remove_punctuation_dict)

        parsed_news.text = parsed_news.text.apply(lambda x: clean(x))
        # Print : Parsed_news are saved in : target
        parsed_news.to_csv(target, sep="\t", index=False)
        return parsed_news

    def parse_behaviors(self, source, target):
        """
        Parse behaviors for using BERT baseline
        Get all the history(NewsID) for each user
        Get the candidate_news from impressions for each user

        Args:
            source: source news file
            target: target news file

        Returns:
            DataFrame: behaviors_parsed(id, history:<NEWS_IDS>, candidate_news<NEWS_IDS>, labels:<y_true>)

        """
        print(f"Parse {source}")
        behaviors = pd.read_table(
            source,
            header=None,
            names=["impression_id", "user", "time", "clicked_news", "impressions"],
        )
        behaviors.clicked_news.fillna(" ", inplace=True)
        parsed_behaviors = pd.DataFrame()
        parsed_behaviors["id"] = behaviors.impression_id
        parsed_behaviors["history"] = behaviors.clicked_news.str.split()
        behaviors.impressions = behaviors.impressions.str.split()
        parsed_behaviors["candidate_news"] = behaviors.impressions.apply(
            lambda x: [impression.split("-")[0] for impression in x])
        parsed_behaviors["labels"] = behaviors.impressions.apply(
            lambda x: [impression.split("-")[1] for impression in x])
        # Print : Parsed_news are saved in : target
        parsed_behaviors.to_csv(target, sep="\t", index=False)

        return parsed_behaviors

    def get_bert_embeddings(self, text, tokenizer, bert):
        encoded_input = tokenizer(text, return_tensors='pt').to(self.device)
        bert = bert.to(self.device)
        output = bert(**encoded_input)
        return output['pooler_output']

    def get_scores(self):
        """
        * for each record:
        - Combine the BERT embedding of the history to get the user representation
        - for each candidate_news:
            = Get the BERT embedding for that news
            = Calculate the cos sim between the user vector and the BERT embedding of that news
            = return the sim
        """
        self.tasks = []
        for index, row in tqdm(self.behaviors_df.iterrows(), desc="Calculating the scores"):
            # Get the user representation
            size = self.get_bert_embeddings("any text to get the shape of the tensor output", self.tokenizer,
                                            self.bert_model).size()
            user = torch.zeros(size).to(self.device)
            for news in row.history:
                user = user + self.get_bert_embeddings(self.news_df.loc[self.news_df['news_id'] == news].text.iat[0],
                                                       self.tokenizer, self.bert_model)
            scores = []
            for news in row.candidate_news:
                news_embedding = self.get_bert_embeddings(self.news_df.loc[self.news_df['news_id'] == news].text.iat[0],
                                                          self.tokenizer, self.bert_model)
                score = cosine_similarity(user.cpu().detach(), news_embedding.cpu().detach())
                scores.append(score)
            # y_pred = [1 if i >= 0.5 else 0 for i in scores]
            y_pred = torch.tensor(np.array(scores))
            y_true = torch.tensor([int(news) for news in row.labels])

            y_pred = torch.reshape(y_pred, (-1,))

            indexes = torch.ones(y_pred.size()) * int(row.id)

            metrics = self.valid_metrics
            metrics_params = self.metrics_params

            for metric, (_, index) in zip(metrics, metrics_params):
                if index:
                    metric.update(preds=y_pred, target=y_true, indexes=indexes.long())
                else:
                    metric.update(preds=y_pred, target=y_true)

    def evaluate(self):

        metrics = self.valid_metrics
        metrics_params = self.metrics_params
        for metric, (name, _) in zip(metrics, metrics_params):
            self.writer.add_scalar(
                f"{name}/valid",
                metric.compute()
            )
            metric.reset()

    def run_pipline(self):
        self.prepare_data()
        self.get_scores()
        self.evaluate()


if __name__ == "__main__":
    model = BERT()
    model.run_pipline()
