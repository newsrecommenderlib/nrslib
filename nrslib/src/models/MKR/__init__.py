import numpy as np
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from .layers import Dense, CrossCompressUnit
import pytorch_lightning as pl
import torchmetrics
from .kGDataset import KGDataset
import inspect

class MKR(pl.LightningModule):
    """MKR: Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation.

    Original paper by:
    Wang, H., Zhang, F., Zhao, M., Li, W., Xie, X., & Guo, M. (2019, May).
    Multi-task feature learning for knowledge graph enhanced recommendation.
    In The world wide web conference (pp. 2000-2010)

    Implementation based on the PyTorch implementation from: https://github.com/hsientzucheng/MKR.PyTorch

    Args:
        dim (int): Dimension for embeddings
        l (int): Number of low layers
        h (int): Number of high layers
        l2_weight (float): Weight of l2 regularization
        lr_rs (float): learning rate for ratings training
        lr_kge (float): Learning rate for knowledge graph training
        kge_interval (int): Knowledge graph embedding training interval
        use_inner_product (bool): Use inner product
        data_dir (str): Data directory
    """
    def __init__(
            self,
            dim,
            l,
            h,
            l2_weight,
            lr_rs,
            lr_kge,
            kge_interval,
            use_inner_product,
            data_dir,
            metrics={},
    ):
        super(MKR, self).__init__()

        np.random.seed(555)

        self.dim = dim
        self.l = l
        self.h = h
        self.l2_weight = l2_weight
        self.lr_rs = lr_rs
        self.lr_kge = lr_kge
        self.kge_interval = kge_interval
        self.use_inner_product = use_inner_product
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.getcwd()

        self.n_item, self.n_user = self._get_n_item_n_user(os.path.join(self.data_dir, r'train'))
        self.kg_dataset = KGDataset(os.path.join(self.data_dir, r'train'))
        self.n_entity = self.kg_dataset.n_entity
        self.n_relation = self.kg_dataset.n_relation
        self.kg = self.kg_dataset.kg

        # Init embeddings
        self.user_embeddings_lookup = nn.Embedding(self.n_user, self.dim)
        self.item_embeddings_lookup = nn.Embedding(self.n_item, self.dim)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.dim)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.dim)

        self.user_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        for i_cnt in range(self.l):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt),
                                     Dense(self.dim, self.dim))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.dim, self.dim))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                    CrossCompressUnit(self.dim))
        # <Higher Model>
        self.kge_pred_mlp = Dense(self.dim * 2, self.dim)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.h - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.dim * 2, self.dim * 2))
        if not self.use_inner_product:
            self.rs_pred_mlp = Dense(self.dim * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.h - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt),
                                       Dense(self.dim * 2, self.dim * 2))

        self._build_model()
        self._build_loss()

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

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
                relation_indices=None, tail_indices=None):

        # <Lower Model>
        if user_indices is not None:
            self.user_indices = user_indices
        if item_indices is not None:
            self.item_indices = item_indices
        if head_indices is not None:
            self.head_indices = head_indices
        if relation_indices is not None:
            self.relation_indices = relation_indices
        if tail_indices is not None:
            self.tail_indices = tail_indices

        # Embeddings
        self.item_embeddings = self.item_embeddings_lookup(self.item_indices)
        self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)
        self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])

        # <Higher Model>
        if user_indices is not None:
            # RS
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            self.user_embeddings = self.user_mlp(self.user_embeddings)
            if self.use_inner_product:
                self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1)
            else:
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1)
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat))
            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]
        if relation_indices is not None:
            # KGE
            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)
            self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1)
            self.head_relation_concat = self.kge_mlp(self.head_relation_concat)
            self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
            self.tail_pred = torch.sigmoid(self.tail_pred)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                                               self.tail_pred, 2), 1) / self.dim))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]

        return outputs


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
        if optimizer_idx == 1:
            if self.current_epoch % self.kge_interval == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss, base_loss_rs, l2_loss_rs, scores, labels = self.train_rs(batch[0])
            return {"loss": loss}

        if optimizer_idx == 1:
            rmse, loss_kge, base_loss_kge, l2_loss_kge = self.train_kge(batch[1])
            return {"loss": loss_kge}

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

    def eval_step(self, batch, batch_idx):
        impressions = batch["impression_id"].long()
        self.eval()
        user_embeddings, item_embeddings, scores, _, labels = self._inference_rs(batch)
        loss, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)
        labels = labels.long()
        return {"loss": loss, "scores": scores, "labels": labels, "impressions": impressions}

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

    def configure_optimizers(self):
        optimizer_rs = torch.optim.Adam(self.parameters(), self.lr_rs)
        optimizer_kge = torch.optim.Adam(self.parameters(), self.lr_kge)
        return optimizer_rs, optimizer_kge

    def _build_model(self):
        print("Build models")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)

    def _build_loss(self):
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()


    def _inference_rs(self, inputs):
        # Inputs
        self.user_indices = inputs['user'].long()
        self.item_indices = inputs['clicked_news'].long()
        labels = inputs['read_status'].float()
        self.head_indices = inputs['clicked_news'].long()

        # Inference
        outputs = self(user_indices=self.user_indices,
                       item_indices=self.item_indices,
                       head_indices=self.head_indices,
                       relation_indices=None,
                       tail_indices=None)

        user_embeddings, item_embeddings, scores, scores_normalized = outputs
        return user_embeddings, item_embeddings, scores, scores_normalized, labels

    def _inference_kge(self, inputs):
        # Inputs
        self.item_indices = inputs['head'].long()
        self.head_indices = inputs['head'].long()
        self.relation_indices = inputs['relation'].long()
        self.tail_indices = inputs['tail'].long()

        # Inference
        outputs = self(user_indices=None,
                       item_indices=self.item_indices,
                       head_indices=self.head_indices,
                       relation_indices=self.relation_indices,
                       tail_indices=self.tail_indices)

        head_embeddings, tail_embeddings, scores_kge, rmse = outputs
        return head_embeddings, tail_embeddings, scores_kge, rmse

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs(self, user_embeddings, item_embeddings, scores, labels):
        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        for name, param in self.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_rs = l2_loss_rs + self.l2_loss(param)
        loss_rs = base_loss_rs + l2_loss_rs * self.l2_weight
        return loss_rs, base_loss_rs, l2_loss_rs

    def loss_kge(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        for name, param in self.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        loss_kge = base_loss_kge + l2_loss_kge * self.l2_weight
        return loss_kge, base_loss_kge, l2_loss_kge

    def train_rs(self, inputs):
        self.train()
        user_embeddings, item_embeddings, scores, _, labels = self._inference_rs(inputs)
        loss_rs, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)

        loss_rs.detach()
        user_embeddings.detach()
        item_embeddings.detach()
        scores.detach()
        labels.detach()

        return loss_rs, base_loss_rs, l2_loss_rs, scores, labels

    def train_kge(self, inputs):
        self.train()
        head_embeddings, tail_embeddings, scores_kge, rmse = self._inference_kge(inputs)
        loss_kge, base_loss_kge, l2_loss_kge = self.loss_kge(scores_kge, head_embeddings, tail_embeddings)

        loss_kge.detach()
        head_embeddings.detach()
        tail_embeddings.detach()
        scores_kge.detach()
        rmse.detach()
        return rmse, loss_kge.sum(), base_loss_kge.sum(), l2_loss_kge

    def _get_n_item_n_user(self, path):
        """Return number of items, users from ratings files

            Args:
                path (str): folder path to the entity2id, user2id files

            Returns:
                Returns a tuple (n_item, n_user) where n_item is the number of unique items and n_user is the number of unique users in the ratings files
        """
        user2int_file = os.path.join(path, r'MKR', r'user2int.txt')
        entity2int_file = os.path.join(path, r'MKR', r'item_index2entity_id_rehashed.txt')

        n_item = 0
        n_user = 0
        with open(user2int_file, encoding="utf-8") as f:
            n_user = len(f.readlines())
        with open(entity2int_file, encoding="utf-8") as f:
            n_item = len(f.readlines())

        return n_item, n_user

