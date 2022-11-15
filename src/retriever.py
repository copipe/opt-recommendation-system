from __future__ import annotations

import gc
import pickle
from abc import ABCMeta, abstractclassmethod
from pathlib import Path
from typing import Dict, List, Tuple

import cudf
import numpy as np
import pandas as pd
import torch
from gensim.models import word2vec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, get_trainer, init_seed
from tqdm import tqdm

from src.metrics import ndcg_score, precision_score, recall_score
from src.utils import flatten_2d_list, order_immutable_deduplication


class Retriever(metaclass=ABCMeta):
    """Abstract class of Retriever (recommendation candidate selection class)

    Attributes:
        top_n (int): Get top n popular items.
        candidate_items (Dict[str, List[str]]): Item list of recommendation candidates searched for by user.
        category_types (List[str]) : List of category types. (A:人材, B:旅行, C:不動産, D:アパレル)
        name (str): Class name.
    """

    def __init__(
        self,
        top_n: int = 10,
    ) -> None:
        self.top_n = top_n
        self.candidate_items: Dict[str, List[str]] = {}
        self.category_types: List[str] = ["A", "B", "C", "D"]
        self.name: str = self.__class__.__name__

    @abstractclassmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractclassmethod
    def search(self, users: List[str]) -> None:
        pass

    def evaluate(
        self,
        label: pd.DataFrame,
        k=22,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:
        """Evaluate the Retriever method.

        Args:
            df (cudf.DataFrame): Preprocessed data.
            k (int): When calculating ndcg, up to the top k are evaluated. Defaults to 22.
            verbose (bool): Verbosity mode.

        Returns:
            Tuple[float, float, float]: (Theoretical max NDCG@K, Recall, Precision)
        """
        # Calculate 3 evaluation indexes (Theoritical max NDCG@K, Recall, Precision)
        n_users = 0
        sum_n_items = 0
        sum_max_ndcg = 0
        sum_recall = 0
        sum_precision = 0
        for user_id, true_scores, true_items in label.values:
            pred_items = self.candidate_items.get(user_id, [])
            true_scores = np.array(true_scores)
            pred_scores = true_scores[np.isin(true_items, pred_items)]

            n_users += 1
            sum_n_items += len(pred_items)
            sum_max_ndcg += ndcg_score(true_scores, pred_scores, k)
            sum_recall += recall_score(true_items, pred_items)
            sum_precision += precision_score(true_items, pred_items)
        n_items = sum_n_items / n_users
        max_ndcg = sum_max_ndcg / n_users
        recall = sum_recall / n_users
        precision = sum_precision / n_users
        if verbose:
            msg = (
                f"[{self.name}] "
                f"n={n_users:,}, n_items={n_items:.1f} "
                f"max_ndcg={max_ndcg:.4f}, recall={recall:.4f}, precision={precision:.4f}"
            )
            print(msg)
        return max_ndcg, recall, precision

    def get_pairs(self, label: pd.DataFrame | None = None) -> pd.DataFrame:
        """Convert self.candidate_items of dictionary type to DataFrame type.

        Args:
            label (pd.DataFrame | None): User action history for the evaluation period. Defaults to None.
            target (bool): Whether to assign a target for ranker. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing all combinations of users and candidate items.
        """
        users = []
        candidates = []
        rank = []
        for user, items in self.candidate_items.items():
            users += [user] * len(items)
            candidates += items
            rank += list(range(len(items)))
        pairs = pd.DataFrame(
            {"user_id": users, "product_id": candidates, f"{self.name}_rank": rank}
        )

        if isinstance(label, pd.DataFrame):
            user_ids = []
            event_types = []
            product_ids = []
            for uid, etype, pid in label.values:
                assert len(etype) == len(pid)
                user_ids += [uid] * len(etype)
                event_types += etype
                product_ids += pid
            pairs_label = pd.DataFrame(
                {
                    "user_id": user_ids,
                    "target": event_types,
                    "product_id": product_ids,
                }
            )
            pairs = pd.merge(
                pairs, pairs_label, how="left", on=["user_id", "product_id"]
            )
            pairs["target"] = pairs["target"].fillna(0).astype(int)
        else:
            pairs["target"] = 0
        return pairs


class PopularItem(Retriever):
    """Select recently popular items as candidate items."""

    def fit(self, df: pd.DataFrame) -> None:
        """Aggregate the recent popular items."""
        # Get top N popular items (items with lots of cv, click, view) for each category.
        df = df[df["event_type"] > 0]

        self.popular_items: Dict[str, List[str]] = {}
        for c in self.category_types:
            vc = df.loc[df["category"] == c, "product_id"].value_counts()
            self.popular_items[c] = vc[: self.top_n].index.tolist()

    def search(self, users: List[str]) -> None:
        """Add popular items to candidate items for each user"""

        for user in users:
            user_category = user.split("_")[1]
            self.candidate_items[user] = self.popular_items[user_category]


class FavoriteItem(Retriever):
    """Select recent favorite items as candidate items."""

    def fit(self, df: pd.DataFrame) -> None:
        """Aggregate the recent favorite items."""
        # Get top N recent favorite items (items with cv, click, view).
        df = df[df["event_type"] > 0]
        df = df.sort_values(["user_id", "time_stamp"], ascending=False)
        df = df.groupby("user_id").agg({"product_id": list})
        df = df.reset_index()

        # Add recent favorite items to candidate items for each user.
        for user_id, items in df.values:
            items = order_immutable_deduplication(items)
            self.candidate_items[user_id] = items[: self.top_n]

    def search(self, users: List[str]) -> None:
        """Add favorite items to candidate items for each user"""
        for user in users:
            if user not in self.candidate_items:
                self.candidate_items[user] = []
        self.candidate_items = {user: self.candidate_items[user] for user in users}


class CoOccurrenceItem(Retriever):
    def __init__(
        self,
        co_occur_items_path: str,
        top_n: int = 10,
    ) -> None:
        super().__init__(top_n)
        self.co_occur_items_path = Path(co_occur_items_path)

    def fit(
        self,
        df: pd.DataFrame,
    ) -> None:

        # Remove other. (cv:3, click:2, view:1, other:0)
        df = df[df["event_type"] > 0]

        if not self.co_occur_items_path.exists():
            self._make_co_occur_items(df)
        else:
            with open(self.co_occur_items_path, "rb") as f:
                self.co_occur_items = pickle.load(f)

        df = df.sort_values(["user_id", "time_stamp"], ascending=False)
        df = df.groupby("user_id").agg({"product_id": list})
        df = df.reset_index()
        for user_id, items in tqdm(df.values):
            n = max(self.top_n // len(items), 20)
            items = order_immutable_deduplication(items)
            items = [self.co_occur_items.get(item, [])[:n] for item in items]
            items = order_immutable_deduplication(flatten_2d_list(items))
            self.candidate_items[user_id] = items[: self.top_n]

    def search(self, users: List[str]) -> None:
        for user in users:
            if user not in self.candidate_items:
                self.candidate_items[user] = []
        self.candidate_items = {user: self.candidate_items[user] for user in users}

    def _make_co_occur_items(self, df: pd.DataFrame) -> None:
        # Convert to cudf for speed.
        df = cudf.from_pandas(df)

        # Extract top_n items that co-occur with each item.
        self.co_occur_items: Dict[str, List[str]] = {}
        items = df["product_id"].unique().to_pandas()
        for item in tqdm(items):
            users = df.loc[df["product_id"] == item]["user_id"].unique()
            df_ = df[(df["user_id"].isin(users)) & (df["product_id"] != item)]
            vc = df_["product_id"].value_counts()
            self.co_occur_items[item] = vc[:1000].to_pandas().index.tolist()

        self.co_occur_items_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.co_occur_items_path, "wb") as f:
            pickle.dump(self.co_occur_items, f)


class RecBoleCF(Retriever):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        config_file_list: List[Path],
        checkpoint_path: str | Path | None = None,
        batch_size: int = 16,
        top_n: int = 10,
    ) -> None:
        super().__init__(top_n)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file_list = config_file_list
        if isinstance(checkpoint_path, str):
            self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.name = f"{self.name}-{model_name}"

    def fit(
        self,
        df: pd.DataFrame,
    ) -> None:

        # train model
        if (self.checkpoint_path is None) or (not self.checkpoint_path.exists()):
            self._train_model()

        # load config from checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        config = checkpoint["config"]
        config.pretrained = True

        # load dataset
        dataset_path = self.checkpoint_path.parent / f"{config.dataset}-dataset.pth"
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        dataset._change_feat_format()

        # load model
        model = get_model(config["model"])(config, dataset)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        model = model.to("cuda")

        user_token2id = dataset.field2token_id["user_id"]
        item_id2token = dataset.field2id_token["item_id"]
        user_ids = df["user_id"].unique().tolist()
        user_tokens = list(map(lambda uid: user_token2id.get(uid, 0), user_ids))
        bs = self.batch_size
        n_iterations = len(user_ids) // bs + 1
        for i in tqdm(range(n_iterations)):
            batch_user_ids = user_ids[i * bs : (i + 1) * bs]
            batch_user_tokens = user_tokens[i * bs : (i + 1) * bs]
            data = Interaction({"user_id": batch_user_tokens})
            data = data.to("cuda")

            # model inference
            pred = model.full_sort_predict(data).view(len(data), -1).to("cuda")
            _, batch_items = torch.topk(pred[:, 1:], self.top_n, dim=1)
            batch_items = batch_items + 1  # TO Remove "[PAD]"
            batch_items = batch_items.to("cpu").detach().numpy()
            batch_items = item_id2token[batch_items].tolist()
            batch_candidates = dict(zip(batch_user_ids, batch_items))
            self.candidate_items.update(batch_candidates)

    def search(self, users: List[str]) -> None:
        for user in users:
            if user not in self.candidate_items:
                self.candidate_items[user] = []
        self.candidate_items = {user: self.candidate_items[user] for user in users}

    def _train_model(self) -> None:
        # configurations initialization
        config = Config(
            model=self.model_name,
            dataset=self.dataset_name,
            config_file_list=self.config_file_list,
        )

        # create dataset
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)

        # dataset splitting
        train_data, valid_data, _ = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config["seed"], config["reproducibility"])
        model = get_model(config["model"])(config, train_data._dataset)
        model.to(config["device"])

        # trainer loading and initialization
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.saved_model_file = str(self.checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # model training
        _ = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config["show_progress"]
        )


class Item2VecCF(Retriever):
    def __init__(
        self,
        model_path: Path,
        top_n: int = 10,
        vector_size: int = 128,
        window: int = 5,
        epochs: int = 10,
        ns_exponent: float = 0.75,
        min_count: int = 5,
        seed: int = 0,
        workers: int = 8,
    ) -> None:
        super().__init__(top_n)
        self.model_path = Path(model_path)
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.ns_exponent = ns_exponent
        self.seed = seed
        self.min_count = min_count
        self.workers = workers

    def fit(
        self,
        df: pd.DataFrame,
    ) -> None:

        df = df.sort_values(["user_id", "time_stamp"], ascending=False)
        df = df.groupby("user_id").agg({"product_id": list})
        df = df.reset_index()

        if not self.model_path.exists():
            self._train_item2vec(df)

        with open(self.model_path, "rb") as f:
            similar_items = pickle.load(f)["similar_items"]

        for user_id, items in tqdm(df.values):
            n = max(self.top_n // len(items), 10)
            items = order_immutable_deduplication(items)
            items = [similar_items.get(item, [])[:n] for item in items]
            items = order_immutable_deduplication(flatten_2d_list(items))
            self.candidate_items[user_id] = items[: self.top_n]

    def search(self, users: List[str]) -> None:
        for user in users:
            if user not in self.candidate_items:
                self.candidate_items[user] = []
        self.candidate_items = {user: self.candidate_items[user] for user in users}

    def _train_item2vec(self, df: pd.DataFrame) -> None:

        item2vec = word2vec.Word2Vec(
            df["product_id"].values.tolist(),
            vector_size=self.vector_size,
            window=self.window,
            epochs=self.epochs,
            ns_exponent=self.ns_exponent,
            seed=self.seed,
            min_count=self.min_count,
            workers=self.workers,
        )

        similar_items: Dict[str, List[str]] = {}
        items = item2vec.wv.index_to_key
        for item in tqdm(items):
            topn_similar_items = item2vec.wv.most_similar(item, topn=200)
            similar_items[item] = [item for item, score in topn_similar_items]

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            result = {
                "model": item2vec,
                "similar_items": similar_items,
            }
            pickle.dump(result, f)


class ConcatRetriever(Retriever):
    def __init__(
        self,
        retrievers: List[Retriever],
        top_n: int = 10,
    ) -> None:
        super().__init__(top_n)
        self.retrievers = retrievers

    def fit(self, df: pd.DataFrame) -> None:

        for retriever in self.retrievers:
            print(f"{retriever.name} fitting...")
            retriever.fit(df)
            gc.collect()
            torch.cuda.empty_cache()

    def search(self, users: List[str]) -> None:
        self.candidate_items = {user: [] for user in users}
        for retriever in self.retrievers:
            retriever.search(users)
            for user in users:
                items = self.candidate_items[user]
                items += retriever.candidate_items[user]
                items = order_immutable_deduplication(items)
                self.candidate_items[user] = items[: self.top_n]

    def evaluate(
        self,
        label: pd.DataFrame,
        k=22,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:

        for retriever in self.retrievers:
            _ = self._evaluate(retriever, label, k, verbose)
            gc.collect()
            torch.cuda.empty_cache()
        max_ndcg, recall, precision = self._evaluate(self, label, k, verbose)
        return max_ndcg, recall, precision

    def _evaluate(
        self,
        retriever: Retriever,
        label: pd.DataFrame,
        k=22,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:
        """Evaluate the Retriever method.

        Args:
            df (cudf.DataFrame): Preprocessed data.
            k (int): When calculating ndcg, up to the top k are evaluated. Defaults to 22.
            verbose (bool): Verbosity mode.

        Returns:
            Tuple[float, float, float]: (Theoretical max NDCG@K, Recall, Precision)
        """
        # Calculate 3 evaluation indexes (Theoritical max NDCG@K, Recall, Precision)
        n_users = 0
        sum_n_items = 0
        sum_max_ndcg = 0
        sum_recall = 0
        sum_precision = 0
        for user_id, true_scores, true_items in label.values:
            pred_items = retriever.candidate_items.get(user_id, [])
            true_scores = np.array(true_scores)
            pred_scores = true_scores[np.isin(true_items, pred_items)]

            n_users += 1
            sum_n_items += len(pred_items)
            sum_max_ndcg += ndcg_score(true_scores, pred_scores, k)
            sum_recall += recall_score(true_items, pred_items)
            sum_precision += precision_score(true_items, pred_items)
        n_items = sum_n_items / n_users
        max_ndcg = sum_max_ndcg / n_users
        recall = sum_recall / n_users
        precision = sum_precision / n_users
        if verbose:
            msg = (
                f"[{retriever.name}] "
                f"n={n_users:,}, n_items={n_items:.1f} "
                f"max_ndcg={max_ndcg:.4f}, recall={recall:.4f}, precision={precision:.4f}"
            )
            print(msg)
        return max_ndcg, recall, precision

    def get_pairs(self, label: pd.DataFrame | None = None) -> pd.DataFrame:
        pairs = self._get_pairs(label)
        for retriever in self.retrievers:
            tmp_pairs = retriever.get_pairs(label).drop(["target"], axis=1)
            pairs = pd.merge(pairs, tmp_pairs, how="left", on=["user_id", "product_id"])
            pairs = pairs.fillna(-1)
            del tmp_pairs
            gc.collect()
        return pairs

    def _get_pairs(self, label: pd.DataFrame | None = None) -> pd.DataFrame:
        users = []
        candidates = []
        for user, items in self.candidate_items.items():
            users += [user] * len(items)
            candidates += items
        pairs = pd.DataFrame({"user_id": users, "product_id": candidates})

        if isinstance(label, pd.DataFrame):
            user_ids = []
            event_types = []
            product_ids = []
            for uid, etype, pid in label.values:
                assert len(etype) == len(pid)
                user_ids += [uid] * len(etype)
                event_types += etype
                product_ids += pid
            pairs_label = pd.DataFrame(
                {
                    "user_id": user_ids,
                    "target": event_types,
                    "product_id": product_ids,
                }
            )
            pairs = pd.merge(
                pairs, pairs_label, how="left", on=["user_id", "product_id"]
            )
            pairs["target"] = pairs["target"].fillna(0).astype(int)
        else:
            pairs["target"] = 0
        return pairs
