from __future__ import annotations

from abc import ABCMeta, abstractclassmethod
from typing import Tuple

import cudf
import numpy as np
import pandas as pd

from src.metrics import ndcg_score


class Retriever(metaclass=ABCMeta):
    def __init__(
        self, df: cudf.DataFrame, date_th: str, train_period: int, eval_period: int = 7
    ) -> None:
        self.df = df
        self.date_th = pd.to_datetime(date_th)
        self.train_start_date = self.date_th - pd.Timedelta(train_period, "days")
        self.eval_end_date = self.date_th + pd.Timedelta(eval_period, "days")
        self.searched_item = {}
        self.category_types = ["A", "B", "C", "D"]

    @abstractclassmethod
    def fit(self) -> None:
        pass

    @abstractclassmethod
    def search(self) -> None:
        pass

    def evaluate(self, k=22) -> Tuple[float, ...]:
        s, e = self.date_th, self.eval_end_date
        df = self.df[
            (s < self.df["time_stamp"]) & (self.df["time_stamp"] <= e)
        ].reset_index(drop=True)
        df = df[df["event_type"] > 0]
        df = df[(df["event_type"] != 3) | (df["ad"] == 1)]
        df = (
            df.groupby(["user_id"])
            .agg({"event_type": list, "product_id": list})
            .reset_index()
        )

        max_ndcgs = []
        recalls = []
        precisions = []
        for user_id, gt_scores, gt_items in df.to_pandas().values:
            pred_items = self.searched_item[user_id]
            pred_scores = gt_scores[np.isin(gt_items, pred_items)]

            max_ndcg = ndcg_score(gt_scores, pred_scores)
            recall = len(set(gt_items) & set(pred_items)) / len(set(gt_items))
            precision = len(set(gt_items) & set(pred_items)) / len(set(pred_items))
            max_ndcgs.append(max_ndcg)
            recalls.append(recall)
            precisions.append(precision)
        return np.mean(max_ndcgs), np.mean(recalls), np.mean(precisions)

    def get_table(self) -> cudf.DataFrame:
        raise NotImplementedError()


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
                self.searched_item[user] = self.popular_items[c]


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
