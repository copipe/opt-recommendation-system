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
from src.utils import flatten_2d_list, order_immutable_deduplication


def get_data_period(
    date_th: str, train_period: int, eval_period: int
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Calculate the first and last days of the train period and evaluation period.

    Args:
        date_th (str): Boundary between train period and evaluation period.
        train_period (int): Number of days of train period.
        eval_period (int, optional): Number of days of evaluation period. Defaults to 7.

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]: First and last day of study period and evaluation period.
    """
    train_start = pd.to_datetime(date_th) - pd.Timedelta(train_period, "days")
    train_end = pd.to_datetime(date_th)
    eval_start = pd.to_datetime(date_th)
    eval_end = pd.to_datetime(date_th) + pd.Timedelta(eval_period, "days")
    return train_start, train_end, eval_start, eval_end


def period_extraction(
    df: cudf.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> cudf.DataFrame:
    """Extract data for specific period.

    Args:
        df (cudf.DataFrame): Preprocessed data.
        start_date (pd.Timestamp): First day of extraction period.
        end_date (pd.Timestamp): Last day of extraction period.

    Returns:
        cudf.DataFrame: Data after extraction.
    """
    df = df[
        (start_date < df["time_stamp"]) & (df["time_stamp"] <= end_date)
    ].reset_index(drop=True)
    return df


class Retriever(metaclass=ABCMeta):
    """Abstract class of Retriever (recommendation candidate selection class)

    Attributes:
        top_n (int): Get top n popular items.
        candidate_items (Dict[str, List[str]]): Item list of recommendation candidates searched for by user.
        category_types (List[str]) : List of category types. (A:人材, B:旅行, C:不動産, D:アパレル)
    """

    def __init__(
        self,
        top_n: int = 10,
    ) -> None:
        self.top_n = top_n
        self.candidate_items: Dict[str, List[str]] = {}
        self.category_types: List[str] = ["A", "B", "C", "D"]

    @abstractclassmethod
    def fit(
        self, df: cudf.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> None:
        pass

    @abstractclassmethod
    def search(self, users: List[str]) -> None:
        pass

    def evaluate(
        self,
        df: cudf.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        k=22,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:
        """Evaluate the Retriever method.

        Args:
            df (cudf.DataFrame): Preprocessed data.
            start_date (pd.Timestamp): First day of evaluation period.
            end_date (pd.Timestamp): Last day of evaluation period.
            k (int): When calculating ndcg, up to the top k are evaluated. Defaults to 22.
            verbose (bool): Verbosity mode.

        Returns:
            Tuple[float, float, float]: (Theoretical max NDCG@K, Recall, Precision)
        """
        # Extract only the data for the evaluation period
        df = period_extraction(df, start_date, end_date)

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
        sum_max_ndcg = 0
        sum_recall = 0
        sum_precision = 0
        for user_id, true_scores, true_items in df.to_pandas().values:
            pred_items = self.candidate_items.get(user_id, [])
            pred_scores = true_scores[np.isin(true_items, pred_items)]

            n_users += 1
            sum_max_ndcg += ndcg_score(true_scores, pred_scores, k)
            sum_recall += recall_score(true_items, pred_items)
            sum_precision += precision_score(true_items, pred_items)
        max_ndcg = sum_max_ndcg / n_users
        recall = sum_recall / n_users
        precision = sum_precision / n_users
        if verbose:
            msg = (
                f"[{self.__class__.__name__}] "
                f"n={n_users:,}, "
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

    def fit(
        self, df: cudf.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> None:
        """Aggregate the recent popular items."""

        # Extract only the data for the train period.
        df = period_extraction(df, start_date, end_date)

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

    def fit(
        self, df: cudf.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> None:
        """Aggregate the recent favorite items."""

        # Extract only the data for the train period.
        df = period_extraction(df, start_date, end_date)

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
    def fit(
        self,
        df: cudf.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        output_path: Path = None,
    ) -> None:

        # Extract only the data for the train period.
        df = period_extraction(df, start_date, end_date)
        # Remove other. (cv:3, click:2, view:1, other:0)
        df = df[df["event_type"] > 0]

        if not output_path.exists():
            # Extract top_n items that co-occur with each item.
            self.co_occur_items: Dict[str, List[str]] = {}
            items = df["product_id"].unique().to_pandas()
            for item in tqdm(items):
                users = df[df["product_id"] == item]["user_id"].unique()
                df_ = df[(df["user_id"].isin(users)) & (df["product_id"] != item)]
                vc = df_["product_id"].value_counts()[: self.top_n]
                self.co_occur_items[item] = vc.index.to_arrow().tolist()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(self.co_occur_items, f)
        else:
            with open(output_path, "rb") as f:
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
        top_n: int = 10,
    ) -> None:
        super().__init__(top_n)
        self.retrievers = retrievers

    def fit(
        self, df: cudf.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> None:

        for retriever in self.retrievers:
            retriever.fit(df, start_date, end_date)

    def search(self, users: List[str]):
        self.candidate_items = {user: [] for user in users}
        for retriever in self.retrievers:
            retriever.search(users)
            for user in users:
                items = self.candidate_items[user]
                items += retriever.candidate_items[user]
                items = order_immutable_deduplication(items)
                self.candidate_items[user] = items
