import torch
import torch.nn as nn
import torch.nn.functional as F
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from ..components.click_predictor.dot_product import DotProductClickPredictor
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import inspect
import torchmetrics


class LSTUR(pl.LightningModule):
    """
    LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self,
                 learning_rate,
                 num_users,
                 num_words,
                 word_embedding_dim,
                 num_categories,
                 num_filters,
                 window_size,
                 query_vector_dim,
                 long_short_term_method,
                 pretrained_word_embedding_path,
                 masking_probability,
                 dropout_probability,
                 metrics={},
                 ):
        """
        # ini
        user embedding: num_filters * 3
        news encoder: num_filters * 3
        GRU:
        input: num_filters * 3
        hidden: num_filters * 3

        # con
        user embedding: num_filter * 1.5
        news encoder: num_filters * 3
        GRU:
        input: num_fitlers * 3
        hidden: num_filter * 1.5
        """
        super(LSTUR, self).__init__()
        try:
            pretrained_word_embedding = torch.from_numpy(
                np.load(pretrained_word_embedding_path)).float()
        except FileNotFoundError:
            pretrained_word_embedding = None
        self.news_encoder = NewsEncoder(num_words, word_embedding_dim, num_categories,
                                        num_filters, window_size, query_vector_dim, dropout_probability,
                                        pretrained_word_embedding)
        self.learning_rate = learning_rate
        self.user_encoder = UserEncoder(num_filters, long_short_term_method)
        self.click_predictor = DotProductClickPredictor()
        self.num_filters = num_filters
        self.num_users = num_users
        self.long_short_term_method = long_short_term_method
        self.masking_probability = masking_probability

        valid_metrics = []
        self.valid_metrics_params = []
        if "valid" in metrics:
            for name, param in metrics["valid"].items():
                d = {}
                target = param.pop("_target_")
                exec(f"metric = {target}(**{str(param)})", globals(), d)
                args = inspect.signature(d["metric"].update).parameters.keys()
                valid_metrics.append(d["metric"])
                self.valid_metrics_params.append((name, "indexes" in args))
        self.valid_metrics = torch.nn.ModuleList(valid_metrics)
        test_metrics = []
        self.test_metrics_params = []
        if "test" in metrics:
            for name, param in metrics["test"].items():
                d = {}
                target = param.pop("_target_")
                exec(f"metric = {target}(**{str(param)})", globals(), d)
                args = inspect.signature(d["metric"].update).parameters.keys()
                test_metrics.append(d["metric"])
                self.test_metrics_params.append((name, "indexes" in args))
        self.test_metrics = torch.nn.ModuleList(test_metrics)

        assert int(self.num_filters * 1.5) == self.num_filters * 1.5
        self.user_embedding = nn.Embedding(
            self.num_users,
            self.num_filters * 3 if self.long_short_term_method == 'ini'
            else int(self.num_filters * 1.5),
            padding_idx=0)

    def forward(self, user, clicked_news_length, candidate_news, clicked_news):
        """
        Args:
            user: batch_size,
            clicked_news_length: batch_size,
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, num_filters * 3
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        # TODO what if not drop
        user = F.dropout2d(self.user_embedding(
            user).unsqueeze(dim=0),
                           p=self.masking_probability,
                           training=self.training).squeeze(dim=0)
        # batch_size, num_clicked_news_a_user, num_filters * 3
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_filters * 3
        user_vector = self.user_encoder(user, clicked_news_length,
                                        clicked_news_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def get_news_vector(self, news):
        # batch_size, num_filters * 3
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user: batch_size
            clicked_news_length: batch_size
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        user = self.user_embedding(user)
        # batch_size, num_filters * 3
        return self.user_encoder(user, clicked_news_length,
                                 clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     self.learning_rate,
                                     )
        return optimizer

    def training_step(self, batch, batch_idx):
        y_pred = self(batch["user"], batch["clicked_news_length"],
                      batch["candidate_news"],
                      batch["clicked_news"])
        y = torch.zeros(len(y_pred)).type_as(y_pred).long()
        loss = F.nll_loss(y_pred, y)

        self.log("step", self.current_epoch, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def eval_step(self, batch, batch_idx):
        for x in batch:
            candidate_news_vector = torch.stack(
                [self.news2vector[news.split("-")[0]] for news in x["impressions"]],
                dim=0,
            )
            user_vector = self.user2vector[x["clicked_news_string"]]
            y_pred = self.get_prediction(candidate_news_vector, user_vector)

            y_true = torch.tensor(
                [int(news.split("-")[1]) for news in x["impressions"]],
                dtype=int,
                device=self.device,
            )

            indexes = (
                    torch.ones(len(y_pred), dtype=int, device=self.device)
                    * x["impression_id"]
            )

            metrics = self.valid_metrics if self.step == "valid" else self.test_metrics
            metrics_params = (
                self.valid_metrics_params
                if self.step == "valid"
                else self.test_metrics_params
            )

            for metric, (_, index) in zip(metrics, metrics_params):
                if index:
                    metric.update(preds=y_pred, target=y_true, indexes=indexes)
                else:
                    metric.update(preds=y_pred, target=y_true)

    def on_eval_start(self):
        news2vector = {}
        news_dataloader = self.trainer.datamodule.news_dataloader(
            self.step, self.device
        )

        for minibatch in tqdm(news_dataloader, desc="Calculating vectors for news"):
            news_ids = minibatch["id"]
            if any(id not in news2vector for id in news_ids):
                news_vector = self.get_news_vector(minibatch)
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector

        news2vector["PADDED_NEWS"] = torch.zeros(list(news2vector.values())[0].size())

        user_dataloader = self.trainer.datamodule.user_dataloader(self.step)

        user2vector = {}
        for minibatch in tqdm(user_dataloader, desc="Calculating vectors for users"):
            user_strings = minibatch["clicked_news_string"]
            if any(user_string not in user2vector for user_string in user_strings):
                clicked_news_vector = torch.stack(
                    [
                        torch.stack(
                            [news2vector[x].to(self.device) for x in news_list], dim=0
                        )
                        for news_list in minibatch["clicked_news"]
                    ],
                    dim=0,
                ).transpose(0, 1)
                user = minibatch["user"].to(self.device)
                clicked_news_length = minibatch["clicked_news_length"].to(self.device)
                user_vector = self.get_user_vector(user, clicked_news_length, clicked_news_vector)
                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector
        self.news2vector = news2vector
        self.user2vector = user2vector
        return super().on_validation_start()

    def eval_epoch_end(self, output):
        metrics = self.valid_metrics if self.step == "valid" else self.test_metrics
        metrics_params = (
            self.valid_metrics_params
            if self.step == "valid"
            else self.test_metrics_params
        )

        self.log(
            "step",
            self.current_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for metric, (name, _) in zip(metrics, metrics_params):
            self.log(
                f"{name}/{self.step}",
                metric.compute(),
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            metric.reset()

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx)

    def on_validation_start(self):
        self.step = "valid"
        self.on_eval_start()

    def validation_epoch_end(self, output):
        self.eval_epoch_end(output)

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx)

    def on_test_start(self):
        self.step = "test"
        self.on_eval_start()

    def on_train_start(self):
        self.step = "train"

    def test_epoch_end(self, output):
        self.eval_epoch_end(output)