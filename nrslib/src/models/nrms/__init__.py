import torch
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from ..components.click_predictor.dot_product import DotProductClickPredictor
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import inspect
import torchmetrics
from tqdm.auto import tqdm


class NRMS(pl.LightningModule):
    """
    NRMS is a neural news recommendation approach with multi-head selfattention. The core of NRMS is a news encoder and a user encoder. In the newsencoder, a multi-head self-attentions is used to learn news representations from news titles by modeling the interactions between words. In the user encoder, we learn representations of users from their browsed news and use multihead self-attention to capture the relatedness between the news. Besides, we apply additive attention to learn more informative news and user representations by selecting important words and news.

    Wu et al. "Neural News Recommendation with Multi-Head Self-Attention." in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)

    Based on implementation from https://github.com/yusanshi/news-recommendation

    Args:
        lr (float): Learning rate
        num_words (int): Number of words
        word_embedding_dim (int): Dimension for word embeddings
        num_attention_heads (int): Number of attention heads
        query_vector_dim (int): Dimension of query vector
        dropout_probability (float): Probability for dropout layer
        pretrained_word_embedding (Optional[str]): Path to pretrained word embeddings
        metrics (dict): Torchmetrics style classes to be used for evaluation
    """

    def __init__(
        self,
        lr,
        num_words,
        word_embedding_dim,
        num_attention_heads,
        query_vector_dim,
        dropout_probability,
        pretrained_word_embedding=None,
        metrics={},
    ):
        super(NRMS, self).__init__()
        self.pretrained_word_embedding = (
            None
            if pretrained_word_embedding is None
            else torch.from_numpy(np.load(pretrained_word_embedding)).float()
        )
        self.lr = lr
        self.news_encoder = NewsEncoder(
            num_words,
            word_embedding_dim,
            num_attention_heads,
            query_vector_dim,
            dropout_probability,
            self.pretrained_word_embedding,
        )
        self.user_encoder = UserEncoder(
            word_embedding_dim, num_attention_heads, query_vector_dim
        )
        self.click_predictor = DotProductClickPredictor()
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
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1
        )
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1
        )
        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector.to(self.device))

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
            news_vector.to(self.device).unsqueeze(dim=0),
            user_vector.to(self.device).unsqueeze(dim=0),
        ).squeeze(dim=0)

    def training_step(self, batch, batch_idx):
        y_pred = self(batch["candidate_news"], batch["clicked_news"])
        y = torch.zeros(len(y_pred)).type_as(y_pred).long()
        loss = F.cross_entropy(y_pred, y)
        self.log(
            "step",
            self.current_epoch,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
