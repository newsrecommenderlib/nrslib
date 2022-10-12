import numpy as np
from os.path import join
import torch
import torch.nn as nn
import os
import collections
import torch.nn.functional as F
import pytorch_lightning as pl
from ...datamodules.mind.preprocessing import prepare_numpy_data
import torchmetrics
from tqdm.auto import tqdm
import inspect


class RippleNet(pl.LightningModule):
    """
    RippleNet: Deep end-to-end model using knowledge graphs for user preference propagation.

    Original paper by:
    Wang, H., Zhang, F., Wang, J., Zhao, M., Li, W., Xie, X., & Guo, M. (2018, October).
    Ripplenet: Propagating user preferences on the knowledge graph for recommender systems.
    In Proceedings of the 27th ACM international conference on information and knowledge management (pp. 417-426).

    Implementation based on the PyTorch RippleNet implementation from: https://github.com/qibinc/RippleNet-PyTorch

    Args:
        dim (int): Dimension for embeddings
        n_hop (int): Maximum number of hops
        kge_weight (float): Knowledge graph weight
        l2_weight (float): Weight of l2 regularization
        lr (float): Learning rate
        n_memory (int): Size of ripple set for each hop
        item_update_mode (str): How to update item at the end of each hop
        using_all_hops (bool): Whether to use outputs of all hops or just the last hop when making predictions
        data_dir (str): Data directory

    """

    def __init__(
        self,
        dim,
        n_hop,
        kge_weight,
        l2_weight,
        lr,
        n_memory,
        item_update_mode,
        using_all_hops,
        data_dir,
        metrics={},
    ):
        super(RippleNet, self).__init__()

        np.random.seed(555)

        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.lr = lr
        self.n_memory = n_memory
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.getcwd()

        self.train_path = join(self.data_dir, "train")
        self.test_path = join(self.data_dir, "test")
        self.valid_path = join(self.data_dir, "valid")

        # ToDO: instantiate in datamodule anjd access them at later point (not init) via self.trainer.datamodule?
        # need to be accessed at later point, not init
        self.user_history_dict = self.prepare_rating_data_train()
        self.n_entity, self.n_relation, self.kg = self.create_knowledge_graph()
        self.ripple_set = self.create_ripple_set()

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.BCELoss()

        self.total_steps_train = 0
        self.total_steps_val = 0
        # ToDo: add self declaration for all args

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

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            h_emb_list.append(self.entity_emb(memories_h[i]))
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings
        )
        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)


    def training_step(self, batch, batch_idx):
        return_dict = self(*self.get_feed_dict(batch))
        loss = return_dict["loss"]
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

    def eval_step(self, batch, batch_idx):
        return_dict = self(*self.get_feed_dict(batch))
        loss = return_dict["loss"]
        items, labels, memories_h, memories_r, memories_t = self.get_feed_dict(batch)
        scores = return_dict["scores"]
        impressions = batch["impression_id"].long()

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
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def prepare_rating_data_train(self):
        """Create a dict containing user ratings
        Returns:
            user_history_dict (dict): user ratings history
        """
        rating_np_train = prepare_numpy_data(self.train_path, "rippleNet")
        user_history_dict = dict()

        i = 0
        while i < len(rating_np_train):
            user = rating_np_train[i][0]
            item = rating_np_train[i][1]
            rating = rating_np_train[i][2]
            if rating == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = []
                user_history_dict[user].append(item)
            i += 1
        return user_history_dict

    def create_knowledge_graph(self):
        """create knowledge graph for use in model training
        Returns:
            n_entity (int): count entities
            n_relation (int): count relations
            kg (ndarray): knowledge graph
        """
        print("creating knowledge graph for model...")
        path_ripple = os.path.join(self.data_dir, r"train", r"rippleNet")

        kg_file = os.path.join(path_ripple, r"kg_final")
        if os.path.exists(kg_file + ".npy"):
            kg_np = np.load(kg_file + ".npy")
        else:
            kg_np = np.loadtxt(kg_file + ".txt", dtype=np.int32)
            np.save(kg_file + ".npy", kg_np)

        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        n_relation = len(set(kg_np[:, 1]))
        kg = collections.defaultdict(list)
        for head, relation, tail in kg_np:
            kg[head].append((tail, relation))
        return n_entity, n_relation, kg

    def create_ripple_set(self):
        """create a ripple set for preference propagation

        Returns:
            ripple_set (defaultdict): contains memories_h, memories_r, memories_t for each user
            memories_h (list): entities that mark the head of knowledge graph relations
            memories_r (list): entities that mark the relation of knowledge graph relations
            memories_t (list): entities that mark the tail of knowledge graph relations
        """
        print("creating ripple set for model...")
        ripple_set = collections.defaultdict(list)
        for user in tqdm(self.user_history_dict):
            for h in range(self.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = self.user_history_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in self.kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                if len(memories_h) == 0:
                    ripple_set[user].append(ripple_set[user][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.n_memory
                    indices = np.random.choice(
                        len(memories_h), size=self.n_memory, replace=replace
                    )
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append((memories_h, memories_r, memories_t))
        return ripple_set

    def get_feed_dict(self, batch):
        """Get Feed dict
        Args:
            batch ():
        Returns:
            items (), labels (), memories_h (list), memories_r (list), memories_t (list)
        """

        items = batch["clicked_news"]
        labels = batch["read_status"]
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.n_hop):
            memories_h.append(
                torch.LongTensor(
                    [self.ripple_set[user.item()][i][0] for user in batch["user"]]
                ).type_as(batch["user"])
            )
            memories_r.append(
                torch.LongTensor(
                    [self.ripple_set[user.item()][i][1] for user in batch["user"]]
                ).type_as(batch["user"])
            )
            memories_t.append(
                torch.LongTensor(
                    [self.ripple_set[user.item()][i][2] for user in batch["user"]]
                ).type_as(batch["user"])
            )
        return items, labels, memories_h, memories_r, memories_t

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2)

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)

        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        return torch.sigmoid(scores)
