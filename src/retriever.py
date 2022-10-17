from __future__ import annotations

import pickle
from abc import ABCMeta, abstractclassmethod
from pathlib import Path
from typing import Dict, List, Tuple

import cudf
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.metrics import ndcg_score, precision_score, recall_score
from src.utils import flatten_2d_list, order_immutable_deduplication, period_extraction


class Retriever(metaclass=ABCMeta):
    """Abstract class of Retriever (recommendation candidate selection class)

    Attributes:
        train_start_date (pd.Timestamp): First day of train period.
        train_end_date (pd.Timestamp): Last day of train period.
        eval_start_date (pd.Timestamp): First day of evaluation period.
        eval_end_date (pd.Timestamp): Last day of evaluation period.
        top_n (int): Get top n popular items.
        candidate_items (Dict[str, List[str]]): Item list of recommendation candidates searched for by user.
        category_types (List[str]) : List of category types. (A:人材, B:旅行, C:不動産, D:アパレル)
        name (str): Class name.
    """

    def __init__(
        self,
        train_start_date: pd.Timestamp,
        train_end_date: pd.Timestamp,
        eval_start_date: pd.Timestamp,
        eval_end_date: pd.Timestamp,
        top_n: int = 10,
    ) -> None:
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.eval_start_date = eval_start_date
        self.eval_end_date = eval_end_date
        self.top_n = top_n
        self.candidate_items: Dict[str, List[str]] = {}
        self.category_types: List[str] = ["A", "B", "C", "D"]
        self.name: str = self.__class__.__name__

    @abstractclassmethod
    def fit(self, df: cudf.DataFrame) -> None:
        pass

    @abstractclassmethod
    def search(self, users: List[str]) -> None:
        pass

    def evaluate(
        self,
        df: cudf.DataFrame,
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
        # Extract only the data for the evaluation period
        df = period_extraction(df, self.eval_start_date, self.eval_end_date)

        # Remove other and (cv=1 and ad !=1). (cv:3, click:2, view:1, other:0)
        df = df[df["event_type"] > 0]
        df = df[(df["event_type"] != 3) | (df["ad"] == 1)]

        # LB's rated users are probably filtered.
        users = df[df["event_type"] > 1]["user_id"].unique()
        df = df[df["user_id"].isin(users)]

        # Extract positive examples for each user.
        df.groupby(["user_id", "product_id"])[
            ["event_type", "time_stamp"]
        ].max().reset_index()
        df = df.sort_values(["user_id", "event_type"], ascending=False)
        df = df.reset_index()
        df = (
            df.groupby(["user_id"])
            .agg({"event_type": list, "product_id": list})
            .reset_index()
        )

        # Calculate 3 evaluation indexes (Theoritical max NDCG@K, Recall, Precision)
        n_users = 0
        sum_n_items = 0
        sum_max_ndcg = 0
        sum_recall = 0
        sum_precision = 0
        for user_id, true_scores, true_items in df.to_pandas().values:
            pred_items = self.candidate_items.get(user_id, [])
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

    def get_table(self) -> cudf.DataFrame:
        """Convert self.candidate_items of dictionary type to DataFrame type.

        Returns:
            cudf.DataFrame: DataFrame containing all combinations of users and candidate items.
        """
        users = []
        candidates = []
        for user, items in self.candidate_items.items():
            users += [user] * len(items)
            candidates += items
        return cudf.DataFrame({"user_id": users, "product_id": candidates})


class PopularItem(Retriever):
    """Select recently popular items as candidate items."""

    def fit(self, df: cudf.DataFrame) -> None:
        """Aggregate the recent popular items."""

        # Extract only the data for the train period.
        df = period_extraction(df, self.train_start_date, self.train_end_date)

        # Get top N popular items (items with lots of cv, click, view) for each category.
        df = df[df["event_type"] > 0]

        self.popular_items: Dict[str, List[str]] = {}
        for c in self.category_types:
            vc = df[df["category"] == c]["product_id"].value_counts()[: self.top_n]
            self.popular_items[c] = vc.to_pandas().index.tolist()

    def search(self, users: List[str]):
        """Add popular items to candidate items for each user"""

        for user in users:
            user_category = user.split("_")[1]
            self.candidate_items[user] = self.popular_items[user_category]


class FavoriteItem(Retriever):
    """Select recent favorite items as candidate items."""

    def fit(self, df: cudf.DataFrame) -> None:
        """Aggregate the recent favorite items."""

        # Extract only the data for the train period.
        df = period_extraction(df, self.train_start_date, self.train_end_date)

        # Get top N recent favorite items (items with cv, click, view).
        df = df[df["event_type"] > 0]
        df = df.sort_values(["user_id", "time_stamp"], ascending=False)
        df = df.groupby("user_id").agg({"product_id": list})
        df = df.reset_index()

        # Add recent favorite items to candidate items for each user.
        for user_id, items in df.to_pandas().values:
            items = order_immutable_deduplication(items.tolist())
            self.candidate_items[user_id] = items[: self.top_n]

    def search(self, users: List[str]):
        """Add favorite items to candidate items for each user"""
        for user in users:
            if user not in self.candidate_items:
                self.candidate_items[user] = []
        self.candidate_items = {user: self.candidate_items[user] for user in users}


class CoOccurrenceItem(Retriever):
    def __init__(
        self,
        output_path: Path,
        train_start_date: pd.Timestamp,
        train_end_date: pd.Timestamp,
        eval_start_date: pd.Timestamp,
        eval_end_date: pd.Timestamp,
        top_n: int = 10,
    ) -> None:
        super().__init__(
            train_start_date, train_end_date, eval_start_date, eval_end_date, top_n
        )
        self.output_path = output_path

    def fit(
        self,
        df: cudf.DataFrame,
    ) -> None:

        # Extract only the data for the train period.
        df = period_extraction(
            df,
            self.train_start_date,
            self.train_end_date,
        )
        # Remove other. (cv:3, click:2, view:1, other:0)
        df = df[df["event_type"] > 0]

        if not self.output_path.exists():
            # Extract top_n items that co-occur with each item.
            self.co_occur_items: Dict[str, List[str]] = {}
            items = df["product_id"].unique().to_pandas()
            for item in tqdm(items):
                users = df[df["product_id"] == item]["user_id"].unique()
                df_ = df[(df["user_id"].isin(users)) & (df["product_id"] != item)]
                vc = df_["product_id"].value_counts()[: self.top_n]
                self.co_occur_items[item] = vc.index.to_arrow().tolist()

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "wb") as f:
                pickle.dump(self.co_occur_items, f)
        else:
            with open(self.output_path, "rb") as f:
                self.co_occur_items = pickle.load(f)

        df = df.sort_values(["user_id", "time_stamp"], ascending=False)
        df = df.groupby("user_id").agg({"product_id": list})
        df = df.reset_index()
        for user_id, items in df.to_pandas().values:
            items = order_immutable_deduplication(items.tolist())[: self.top_n]
            items = [self.co_occur_items.get(item, []) for item in items]
            self.candidate_items[user_id] = flatten_2d_list(items)[: self.top_n]

    def search(self, users: List[str]):
        for user in users:
            if user not in self.candidate_items:
                self.candidate_items[user] = []
        self.candidate_items = {user: self.candidate_items[user] for user in users}


class ConcatRetriever(Retriever):
    def __init__(
        self,
        retrievers: List[Retriever],
        train_start_date: pd.Timestamp,
        train_end_date: pd.Timestamp,
        eval_start_date: pd.Timestamp,
        eval_end_date: pd.Timestamp,
        top_n: int = 10,
    ) -> None:
        super().__init__(
            train_start_date, train_end_date, eval_start_date, eval_end_date, top_n
        )
        self.retrievers = retrievers

    def fit(self, df: cudf.DataFrame) -> None:

        for retriever in self.retrievers:
            retriever.fit(df)

    def search(self, users: List[str]):
        self.candidate_items = {user: [] for user in users}
        for retriever in self.retrievers:
            retriever.search(users)
            for user in users:
                items = self.candidate_items[user]
                items += retriever.candidate_items[user]
                items = order_immutable_deduplication(items)
                self.candidate_items[user] = items[: self.top_n]
