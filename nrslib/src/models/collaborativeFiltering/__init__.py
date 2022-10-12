from os.path import join
import os
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable
import inspect



class CollaborativeFiltering(pl.LightningModule):
    """Neural Collaborative Filtering Model.

    Models user-news article interactions through a neural network.

    Implementation based on the PyTorch implementation from: https://github.com/andreas-bauer/ml-movie-recommender

    Args:
        lr (float): learning rate
        n_factors (int): number of factors for embeddings
        data_dir (str): data directory
    """
    def __init__(
        self,
        lr,
        n_factors,
        data_dir,
        metrics={},
    ):
        super(CollaborativeFiltering, self).__init__()
        self.lr = lr
        self.n_factors = n_factors
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.getcwd()

        self.train_path = join(self.data_dir, "train")
        self.test_path = join(self.data_dir, "test")
        self.valid_path = join(self.data_dir, "valid")
        self.n_users, self.n_news = self.get_n()

        self.users = nn.Embedding(self.n_users, self.n_factors)
        self.news = nn.Embedding(self.n_news, self.n_factors)
        self.users.weight.data.uniform_(0, 0.5)
        self.news.weight.data.uniform_(0, 0.5)

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

    def eval_step(self, batch, batch_idx):
        users = batch["user"]
        news = batch["clicked_news"]
        targ = Variable(batch["read_status"].float())
        scores = self(users, news)
        labels = batch["read_status"]
        users = batch["user"].long()
        impressions = batch["impression_id"].long()

        criterion = nn.MSELoss()
        loss = criterion(scores, targ)

        return {"loss": loss, "scores": scores, "labels": labels, "users": users , "impressions": impressions}

    def eval_step_end(self, outputs):
        loss = outputs["loss"]
        scores = outputs["scores"]
        labels = outputs["labels"]
        impressions = outputs["impressions"]

        metrics = self.valid_metrics if self.step == "valid" else self.test_metrics
        metrics_params = (
            self.valid_metrics_params
            if self.step == "valid"
            else self.test_metrics_params
        )

        for metric, (_, index) in zip(metrics, metrics_params):
            if index:
                metric.update(preds=scores, target=labels, indexes=impressions)
            else:
                metric.update(preds=scores, target=labels)

        return {'loss': loss}

    def eval_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("step", torch.tensor(self.current_epoch).to(torch.float32))
        self.log("loss/" + self.step, avg_loss, logger=True, prog_bar=True)
        metrics = self.valid_metrics if self.step == "valid" else self.test_metrics
        metrics_params = (
            self.valid_metrics_params
            if self.step == "valid"
            else self.test_metrics_params
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

    def training_step(self, batch, batch_idx):
        users = batch["user"]
        news = batch["clicked_news"]
        targ = Variable(batch["read_status"].float())
        scores = self(users, news)
        criterion = nn.MSELoss()
        loss = criterion(scores, targ)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.step = "valid"
        return self.eval_step(batch, batch_idx)

    def validation_step_end(self, outputs):
        return self.eval_step_end(outputs)

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        self.step = "test"
        return self.eval_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.eval_step_end(outputs)

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs)

    def forward(self, u, n):
        return (self.users(u) * self.news(n)).sum(1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def get_n(self):
        """Get total number of users and news articles.

        Returns:
            Number of users and number of news.

        """
        n_users = 0
        n_news = 0
        with open(os.path.join(self.train_path, "collaborative", "item_index2entity_id_rehashed.txt"), "r") as f:
            for line in f:
                n_news += 1
        with open(os.path.join(self.train_path, "collaborative", "user2int.txt"), "r") as f:
            for line in f:
                n_users += 1
        return n_users, n_news
