from abc import ABCMeta, abstractmethod
from typing import List

import cudf
import numpy as np
import pandas as pd

from src.utils import period_extraction


class AbstractFeatureTransformer(metaclass=ABCMeta):
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.name = self.__class__.__name__

    def fit_transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
        self.fit(df, pairs)
        return self.transform(df, pairs)

    @abstractmethod
    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
        pass


class UserActionScore(AbstractFeatureTransformer):
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        decay_rate: int = 1.0,
    ):
        super().__init__(start_date, end_date)
        self.decay_rate = decay_rate

    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
        pass

    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
        df = period_extraction(df, self.start_date, self.end_date)
        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
        decay_rate = self.decay_rate
        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)

        action_list = [
            ("cv", 3),
            ("click", 2),
            ("pv", 1),
            ("other", 0),
        ]
        pairs["index"] = np.arange(len(pairs))
        new_feature_names = []
        for event_name, event_type in action_list:
            feature_name = f"{event_name}-score-r{self.decay_rate}_by_user"
            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
            feature = df.groupby("user_id")["score"].sum().reset_index()
            feature = feature.rename(columns={"score": feature_name})

            pairs = cudf.merge(pairs, feature, how="left", on="user_id")
            pairs[feature_name] = pairs[feature_name].fillna(0)
            new_feature_names.append(feature_name)
        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
        return pairs[new_feature_names]


class ItemActionScore(AbstractFeatureTransformer):
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        decay_rate: int = 1.0,
    ):
        super().__init__(start_date, end_date)
        self.decay_rate = decay_rate

    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
        pass

    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
        df = period_extraction(df, self.start_date, self.end_date)
        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
        decay_rate = self.decay_rate
        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)

        action_list = [
            ("cv", 3),
            ("click", 2),
            ("pv", 1),
            ("other", 0),
        ]
        pairs["index"] = np.arange(len(pairs))
        new_feature_names = []
        for event_name, event_type in action_list:
            feature_name = f"{event_name}-score-r{self.decay_rate}_by_item"
            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
            feature = df.groupby("product_id")["score"].sum().reset_index()
            feature = feature.rename(columns={"score": feature_name})

            pairs = cudf.merge(pairs, feature, how="left", on="product_id")
            pairs[feature_name] = pairs[feature_name].fillna(0)
            new_feature_names.append(feature_name)
        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
        return pairs[new_feature_names]


class UserItemActionScore(AbstractFeatureTransformer):
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        decay_rate: int = 1.0,
    ):
        super().__init__(start_date, end_date)
        self.decay_rate = decay_rate

    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
        pass

    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
        df = period_extraction(df, self.start_date, self.end_date)
        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
        decay_rate = self.decay_rate
        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)

        action_list = [
            ("cv", 3),
            ("click", 2),
            ("pv", 1),
            ("other", 0),
        ]
        pairs["index"] = np.arange(len(pairs))
        new_feature_names = []
        for event_name, event_type in action_list:
            feature_name = f"{event_name}-score-r{self.decay_rate}_by_user-item"
            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
            feature = df.groupby(["user_id", "product_id"])["score"].sum().reset_index()
            feature = feature.rename(columns={"score": feature_name})

            pairs = cudf.merge(pairs, feature, how="left", on=["user_id", "product_id"])
            pairs[feature_name] = pairs[feature_name].fillna(0)
            new_feature_names.append(feature_name)
        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
        return pairs[new_feature_names]


class ConcatFeatureTransformer(AbstractFeatureTransformer):
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        feature_transformers: List[AbstractFeatureTransformer],
    ):
        super().__init__(start_date, end_date)
        self.feature_transformers = feature_transformers

    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
        for feature_transformer in self.feature_transformers:
            feature_transformer.fit(df, pairs)

    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
        features = []
        for feature_transformer in self.feature_transformers:
            features.append(feature_transformer.transform(df, pairs))
        return cudf.concat(features, axis=1)
