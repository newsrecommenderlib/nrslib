import torch
import torch.nn as nn
from .KCNN import KCNN
from .attention import Attention
from ..components.click_predictor.DNN import DNNClickPredictor
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from tqdm import tqdm
import inspect

class DKN(pl.LightningModule):
    """
    Deep knowledge-aware network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self,
                 learning_rate,
                 num_words,
                 word_embedding_dim,
                 use_context,
                 num_entities,
                 entity_embedding_dim,
                 num_filters,
                 window_sizes,
                 query_vector_dim,
                 num_clicked_news_a_user,
                 pretrained_word_embedding_path,
                 pretrained_entity_embedding_path,
                 pretrained_context_embedding_path,
                metrics={}):
        super(DKN, self).__init__()
        try:
            pretrained_word_embedding = torch.from_numpy(
                np.load(pretrained_word_embedding_path)).float()
        except FileNotFoundError:
            pretrained_word_embedding = None
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(pretrained_entity_embedding_path)).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None
        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(pretrained_context_embedding_path)).float()
        except FileNotFoundError:  # TODO Throw error when not found
            pretrained_context_embedding = None
        self.learning_rate = learning_rate
        self.window_sizes = window_sizes
        self.num_filters = num_filters
        self.num_clicked_news_a_user = num_clicked_news_a_user
        self.kcnn = KCNN(pretrained_word_embedding,
                         pretrained_entity_embedding,
                         pretrained_context_embedding,
                         num_words,
                         word_embedding_dim,
                         use_context,
                         num_entities,
                         entity_embedding_dim, num_filters, window_sizes, query_vector_dim)
        self.attention = Attention(self.window_sizes, self.num_filters, self.num_clicked_news_a_user)
        self.click_predictor = DNNClickPredictor(
            len(self.window_sizes) * 2 * self.num_filters)

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

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": batch_size * num_words_title,
                        "title_entities": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title": batch_size * num_words_title,
                        "title_entities": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, len(window_sizes) * num_filters
        candidate_news_vector = torch.stack(
            [self.kcnn(x) for x in candidate_news], dim=1)
        # batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        clicked_news_vector = torch.stack([self.kcnn(x) for x in clicked_news],
                                          dim=1)
        # batch_size, 1 + K, len(window_sizes) * num_filters
        user_vector = torch.stack([
            self.attention(x, clicked_news_vector)
            for x in candidate_news_vector.transpose(0, 1)
        ],
            dim=1)
        size = candidate_news_vector.size()
        # batch_size, 1 + K
        click_probability = self.click_predictor(
            candidate_news_vector.view(size[0] * size[1], size[2]),
            user_vector.view(size[0] * size[1],
                             size[2])).view(size[0], size[1])
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title,
                    "title_entities": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, len(window_sizes) * num_filters
        """
        # batch_size, len(window_sizes) * num_filters
        return self.kcnn(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            (shape) batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        """
        # batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        return clicked_news_vector

    def get_prediction(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: candidate_size, len(window_sizes) * num_filters
            clicked_news_vector: num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            click_probability: 0-dim tensor
        """
        # candidate_size, len(window_sizes) * num_filters
        user_vector = self.attention(candidate_news_vector,
                                     clicked_news_vector.expand(candidate_news_vector.size(0), -1, -1))
        # candidate_size
        click_probability = self.click_predictor(candidate_news_vector, user_vector)

        return click_probability

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     self.learning_rate,
                                     )
        return optimizer

    def training_step(self, batch, batch_idx):
        y_pred = self(batch["candidate_news"], batch["clicked_news"])
        y = torch.zeros(len(y_pred)).type_as(y_pred).long()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y)
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

                user_vector = self.get_user_vector(clicked_news_vector)
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
