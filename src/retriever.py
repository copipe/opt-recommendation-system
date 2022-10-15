from __future__ import annotations

from abc import ABCMeta, abstractclassmethod
from typing import Dict, List, Tuple

import cudf
import numpy as np
import pandas as pd

from src.metrics import ndcg_score, precision_score, recall_score


class Retriever(metaclass=ABCMeta):
    """Abstract class of Retriever (recommendation candidate selection class)

    Attributes:
        df (cudf.DataFrame): Preprocessed data.
        date_th (str): Boundary between train period and evaluation period.
        train_period (int): Number of days of train period.
        eval_period (int, optional): Number of days of evaluation period. Defaults to 7.
        candidate_items (Dict[str, List[str]]): Item list of recommendation candidates searched for by user.
        category_types (List[str]) : List of category types. (A:人材, B:旅行, C:不動産, D:アパレル)
    """

    def __init__(
        self, df: cudf.DataFrame, date_th: str, train_period: int, eval_period: int = 7
    ) -> None:
        self.df = df
        self.date_th = pd.to_datetime(date_th)
        self.train_start_date = self.date_th - pd.Timedelta(train_period, "days")
        self.eval_end_date = self.date_th + pd.Timedelta(eval_period, "days")
        self.candidate_items: Dict[str, List[str]] = {}
        self.category_types: List[str] = ["A", "B", "C", "D"]

    @abstractclassmethod
    def fit(self) -> None:
        pass

    @abstractclassmethod
    def search(self) -> None:
        pass

    def evaluate(self, k=22, verbose: bool = True) -> Tuple[float, float, float]:
        """Evaluate the Retriever method.

        Args:
            k (int): When calculating ndcg, up to the top k are evaluated. Defaults to 22.

        Returns:
            Tuple[float, float, float]: (Theoretical max NDCG@K, Recall, Precision)
            verbose (bool): Verbosity mode.
        """
        # Extract only the data for the evaluation period
        s, e = self.date_th, self.eval_end_date
        df = self.df[
            (s < self.df["time_stamp"]) & (self.df["time_stamp"] <= e)
        ].reset_index(drop=True)

        # Remove other and (cv=1 and ad !=1). (cv:3, click:2, view:1, other:0)
        df = df[df["event_type"] > 0]
        df = df[(df["event_type"] != 3) | (df["ad"] == 1)]

        # Extract positive examples for each user.
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
            pred_items = self.candidate_items[user_id]
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
    def __init__(
        self,
        df: cudf.DataFrame,
        date_th: str,
        train_period: int,
        eval_period: int = 7,
        top_n: int = 10,
    ):
        super().__init__(df, date_th, train_period, eval_period)
        self.popular_items = {}
        self.top_n = top_n

    def fit(self) -> None:
        s, e = self.train_start_date, self.date_th
        df = self.df[
            (s < self.df["time_stamp"]) & (self.df["time_stamp"] <= e)
        ].reset_index(drop=True)

        for c in self.category_types:
            vc = df[df["category"] == c]["product_id"].value_counts()[: self.top_n]
            self.popular_items[c] = vc.to_pandas().index.tolist()

    def search(self) -> None:
        for c in self.category_types:
            users = self.df[self.df["category"] == c]["user_id"].to_pandas().unique()
            for user in users:
                self.candidate_items[user] = self.popular_items[c]


class PastInterest(Retriever):
    def __init__(self):
        super().__init__()

    def fit(self) -> None:
        raise NotImplementedError()

    def search(self) -> None:
        raise NotImplementedError()


class CoOccurrenceItem(Retriever):
    def __init__(self):
        super().__init__()

    def fit(self) -> None:
        raise NotImplementedError()

    def search(self) -> None:
        raise NotImplementedError()
