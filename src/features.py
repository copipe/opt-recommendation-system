from __future__ import annotations

import gc
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from recbole.data.interaction import Interaction
from recbole.utils import get_model
from tqdm import tqdm


class AbstractFeatureTransformer(metaclass=ABCMeta):
    def __init__(self, save_file_path: str):
        self.name = self.__class__.__name__
        self.save_file_path = Path(save_file_path)

    def fit_transform(self, df: pd.DataFrame, pairs: pd.DataFrame):
        self.fit(df, pairs)
        return self.transform(df, pairs)

    @abstractmethod
    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        pass

    def _save_feature(self, df: pd.DataFrame) -> None:
        self.save_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(self.save_file_path)

    def _load_feature(self) -> pd.DataFrame:
        return pd.read_pickle(self.save_file_path)


class RecentActionFrequency(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
        decay_rate: int = 1.0,
    ):
        super().__init__(save_file_path)
        self.decay_rate = decay_rate

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            action_list = [
                ("cv", 3),
                ("click", 2),
                ("pv", 1),
                ("other", 0),
            ]

            groupby_keys = [
                "user_id",
                "product_id",
                ["user_id", "product_id"],
            ]

            end_date = df["time_stamp"].max()
            decay_rate = self.decay_rate
            df["day_diff"] = (end_date - df["time_stamp"]) / np.timedelta64(1, "D")
            df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)

            new_feature_names = []
            for gkey in groupby_keys:
                str_gkey = "-".join(gkey) if isinstance(gkey, list) else gkey
                for event_name, event_type in action_list:
                    feature_name = (
                        f"{event_name}-score-r{self.decay_rate}_by_{str_gkey}"
                    )
                    df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
                    feature = df.groupby(gkey)["score"].sum().reset_index()
                    feature = feature.rename(columns={"score": feature_name})
                    pairs = pd.merge(pairs, feature, how="left", on=gkey)
                    pairs[feature_name] = pairs[feature_name].fillna(0)
                    new_feature_names.append(feature_name)
            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class ItemAttribute(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
    ):
        super().__init__(save_file_path)

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            pairs["item_category"] = pairs["product_id"].str.split("_", expand=True)[1]
            new_feature_names = ["item_category"]
            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class RecBolePredictor(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
        model_name: str,
        dataset_name: str,
        config_file_list: List[Path],
        checkpoint_path: str | Path | None = None,
        batch_size: int = 16,
    ):
        super().__init__(save_file_path)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file_list = config_file_list
        if isinstance(checkpoint_path, str):
            self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.name = f"{self.name}-{model_name}"

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.checkpoint_path.exists():
                raise "{str(self.checkpoint_path)} is not exist."

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
            item_token2id = dataset.field2token_id["item_id"]
            user_ids = list(
                map(lambda uid: user_token2id.get(uid, 0), pairs["user_id"])
            )
            item_ids = list(
                map(lambda pid: item_token2id.get(pid, 0), pairs["product_id"])
            )

            bs = self.batch_size
            n_iterations = len(user_ids) // bs + 1
            pred = []
            for i in tqdm(range(n_iterations)):
                batch_user_tokens = user_ids[i * bs : (i + 1) * bs]
                batch_item_tokens = item_ids[i * bs : (i + 1) * bs]
                data = Interaction(
                    {
                        "user_id": batch_user_tokens,
                        "item_id": batch_item_tokens,
                    }
                )
                data = data.to("cuda")

                # model inference
                batch_pred = model.predict(data)
                pred += batch_pred.to("cpu").detach().tolist()

            feature_name = f"{self.model_name}_pred"
            pairs[feature_name] = pred

            new_feature_names = [feature_name]
            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class ConcatFeatureTransformer(AbstractFeatureTransformer):
    def __init__(
        self,
        feature_transformers: List[AbstractFeatureTransformer],
        save_file_path="default.pickle",
    ):
        super().__init__(save_file_path)
        self.feature_transformers = feature_transformers

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        for feature_transformer in self.feature_transformers:
            feature_transformer.fit(df, pairs)

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        features = []
        for feature_transformer in self.feature_transformers:
            features.append(feature_transformer.transform(df, pairs))
            gc.collect()
            torch.cuda.empty_cache()
        return pd.concat(features, axis=1)


# class UserActionScore(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        decay_rate: int = 1.0,
#    ):
#        super().__init__()
#        self.decay_rate = decay_rate
#
#    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
#        pass
#
#    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
#        end_date = df["time_stamp"].max()
#        decay_rate = self.decay_rate
#        df["day_diff"] = (end_date - df["time_stamp"]) / np.timedelta64(1, "D")
#        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)
#
#        action_list = [
#            ("cv", 3),
#            ("click", 2),
#            ("pv", 1),
#            ("other", 0),
#        ]
#        pairs["index"] = np.arange(len(pairs))
#        new_feature_names = []
#        for event_name, event_type in action_list:
#            feature_name = f"{event_name}-score-r{self.decay_rate}_by_user"
#            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
#            feature = df.groupby("user_id")["score"].sum().reset_index()
#            feature = feature.rename(columns={"score": feature_name})
#
#            pairs = pd.merge(pairs, feature, how="left", on="user_id")
#            pairs[feature_name] = pairs[feature_name].fillna(0)
#            new_feature_names.append(feature_name)
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]
#
#
# class ItemActionScore(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        decay_rate: int = 1.0,
#    ):
#        super().__init__(start_date, end_date)
#        self.decay_rate = decay_rate
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#        df = period_extraction(df, self.start_date, self.end_date)
#        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
#        decay_rate = self.decay_rate
#        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)
#
#        action_list = [
#            ("cv", 3),
#            ("click", 2),
#            ("pv", 1),
#            ("other", 0),
#        ]
#        pairs["index"] = np.arange(len(pairs))
#        new_feature_names = []
#        for event_name, event_type in action_list:
#            feature_name = f"{event_name}-score-r{self.decay_rate}_by_item"
#            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
#            feature = df.groupby("product_id")["score"].sum().reset_index()
#            feature = feature.rename(columns={"score": feature_name})
#
#            pairs = cudf.merge(pairs, feature, how="left", on="product_id")
#            pairs[feature_name] = pairs[feature_name].fillna(0)
#            new_feature_names.append(feature_name)
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]
#
#
# class UserItemActionScore(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        decay_rate: int = 1.0,
#    ):
#        super().__init__(start_date, end_date)
#        self.decay_rate = decay_rate
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#        df = period_extraction(df, self.start_date, self.end_date)
#        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
#        decay_rate = self.decay_rate
#        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)
#
#        action_list = [
#            ("cv", 3),
#            ("click", 2),
#            ("pv", 1),
#            ("other", 0),
#        ]
#        pairs["index"] = np.arange(len(pairs))
#        new_feature_names = []
#        for event_name, event_type in action_list:
#            feature_name = f"{event_name}-score-r{self.decay_rate}_by_user-item"
#            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
#            feature = df.groupby(["user_id", "product_id"])["score"].sum().reset_index()
#            feature = feature.rename(columns={"score": feature_name})
#
#            pairs = cudf.merge(pairs, feature, how="left", on=["user_id", "product_id"])
#            pairs[feature_name] = pairs[feature_name].fillna(0)
#            new_feature_names.append(feature_name)
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]


# class UserAttribute(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        model_dir: str,
#    ):
#        super().__init__(start_date, end_date)
#
#        self.category_types: List[str] = ["A", "B", "C", "D"]
#        self.item2vec = {}
#        self.kmeans = {}
#        for category in self.category_types:
#            with open(model_dir / f"item2vec_{category}.pickle", "rb") as f:
#                self.item2vec[category] = pickle.load(f)
#
#            with open(model_dir / f"kmeans_{category}.pickle", "rb") as f:
#                self.kmeans[category] = pickle.load(f)
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#
#        df = period_extraction(df, self.start_date, self.end_date)
#
#        user_ids = []
#        user_cluster_ids = []
#        for category in self.category_types:
#
#            df_ = df[df["category"] == category]
#            df_ = df_.sort_values(
#                ["user_id", "event_type", "time_stamp"], ascending=False
#            )
#            df_ = df_.groupby("user_id").agg({"product_id": list})
#            df_ = df_.reset_index()
#
#            item2vec = self.item2vec[category]
#            kmeans = self.kmeans[category]
#            user_vectors = []
#            for user_id, items in df_.to_pandas().values:
#                items = order_immutable_deduplication(list(items))[:10]
#                user_vector = item2vec.wv.get_mean_vector(items)
#                user_vectors.append(user_vector)
#                user_ids.append(user_id)
#            user_cluster_ids += list(kmeans.predict(np.array(user_vectors)))
#
#        user_cluster = cudf.DataFrame(
#            {
#                "user_id": user_ids,
#                "user_cluster_id": user_cluster_ids,
#            }
#        )
#
#        pairs["index"] = np.arange(len(pairs))
#        pairs = cudf.merge(pairs, user_cluster, how="left", on="user_id")
#        pairs["user_cluster_id"] = pairs["user_cluster_id"].fillna(-1)
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        new_feature_names = ["user_cluster_id"]
#        return pairs[new_feature_names]
#
#
# class WeeklyActionSimilarity(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        decay_rate: int = 1.0,
#    ):
#        super().__init__(start_date, end_date)
#        self.decay_rate = decay_rate
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#        df = period_extraction(df, self.start_date, self.end_date)
#        df["week"] = df["time_stamp"].dt.weekday
#        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
#        decay_rate = self.decay_rate
#        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)
#
#        action_list = [
#            ("cv", 3),
#            ("click", 2),
#            ("pv", 1),
#            ("other", 0),
#        ]
#        pairs["index"] = np.arange(len(pairs))
#        new_feature_names = []
#        for event_name, event_type in action_list:
#            weekly_user_feature_names = []
#            weekly_item_feature_names = []
#            for week in range(7):
#                df["score"] = (
#                    df["weight_decay"]
#                    * (df["event_type"] == event_type)
#                    * (df["week"] == week)
#                )
#
#                # user feature
#                feature_name = (
#                    f"week{week}_{event_name}-score-r{self.decay_rate}_by_user"
#                )
#                feature = df.groupby("user_id")["score"].sum().reset_index()
#                feature = feature.rename(columns={"score": feature_name})
#                pairs = cudf.merge(pairs, feature, how="left", on="user_id")
#                pairs[feature_name] = pairs[feature_name].fillna(0)
#                weekly_user_feature_names.append(feature_name)
#
#                # item feature
#                feature_name = (
#                    f"week{week}_{event_name}-score-r{self.decay_rate}_by_item"
#                )
#                feature = df.groupby("product_id")["score"].sum().reset_index()
#                feature = feature.rename(columns={"score": feature_name})
#                pairs = cudf.merge(pairs, feature, how="left", on="product_id")
#                pairs[feature_name] = pairs[feature_name].fillna(0)
#                weekly_item_feature_names.append(feature_name)
#
#            # similarity
#            feature_name = f"week-similarity_{event_name}-score-r{self.decay_rate}"
#            user_vectors = pairs[weekly_user_feature_names].values
#            item_vectors = pairs[weekly_item_feature_names].values
#            user_norm = np.linalg.norm(user_vectors, axis=1, keepdims=True)
#            item_norm = np.linalg.norm(item_vectors, axis=1, keepdims=True)
#            user_norm[user_norm == 0] = 1
#            item_norm[item_norm == 0] = 1
#            pairs[feature_name] = (
#                (user_vectors / user_norm) * (item_vectors / item_norm)
#            ).sum(axis=1)
#            pairs = pairs.drop(weekly_user_feature_names, axis=1)
#            pairs = pairs.drop(weekly_item_feature_names, axis=1)
#            new_feature_names.append(feature_name)
#
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]
#
#
# class TimelyActionSimilarity(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        decay_rate: int = 1.0,
#    ):
#        super().__init__(start_date, end_date)
#        self.decay_rate = decay_rate
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#        df = period_extraction(df, self.start_date, self.end_date)
#        df["time"] = (df["time_stamp"].dt.hour) // 4
#        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
#        decay_rate = self.decay_rate
#        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)
#
#        action_list = [
#            ("cv", 3),
#            ("click", 2),
#            ("pv", 1),
#            ("other", 0),
#        ]
#        pairs["index"] = np.arange(len(pairs))
#        new_feature_names = []
#        for event_name, event_type in action_list:
#            timely_user_feature_names = []
#            timely_item_feature_names = []
#            for time in range(6):
#                df["score"] = (
#                    df["weight_decay"]
#                    * (df["event_type"] == event_type)
#                    * (df["time"] == time)
#                )
#
#                # user feature
#                feature_name = (
#                    f"time{time}_{event_name}-score-r{self.decay_rate}_by_user"
#                )
#                feature = df.groupby("user_id")["score"].sum().reset_index()
#                feature = feature.rename(columns={"score": feature_name})
#                pairs = cudf.merge(pairs, feature, how="left", on="user_id")
#                pairs[feature_name] = pairs[feature_name].fillna(0)
#                timely_user_feature_names.append(feature_name)
#
#                # item feature
#                feature_name = (
#                    f"time{time}_{event_name}-score-r{self.decay_rate}_by_item"
#                )
#                feature = df.groupby("product_id")["score"].sum().reset_index()
#                feature = feature.rename(columns={"score": feature_name})
#                pairs = cudf.merge(pairs, feature, how="left", on="product_id")
#                pairs[feature_name] = pairs[feature_name].fillna(0)
#                timely_item_feature_names.append(feature_name)
#
#            # similarity
#            feature_name = f"time-similarity_{event_name}-score-r{self.decay_rate}"
#            user_vectors = pairs[timely_user_feature_names].values
#            item_vectors = pairs[timely_item_feature_names].values
#            user_norm = np.linalg.norm(user_vectors, axis=1, keepdims=True)
#            item_norm = np.linalg.norm(item_vectors, axis=1, keepdims=True)
#            user_norm[user_norm == 0] = 1
#            item_norm[item_norm == 0] = 1
#            pairs[feature_name] = (
#                (user_vectors / user_norm) * (item_vectors / item_norm)
#            ).sum(axis=1)
#            pairs = pairs.drop(timely_user_feature_names, axis=1)
#            pairs = pairs.drop(timely_item_feature_names, axis=1)
#            new_feature_names.append(feature_name)
#
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]
#
#
# class ItemGroupActionScore(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        item_cluster_path: str,
#        decay_rate: float,
#    ):
#        super().__init__(start_date, end_date)
#        self.item_cluster = cudf.read_csv(item_cluster_path)
#        self.decay_rate = decay_rate
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#        df = cudf.merge(df, self.item_cluster, how="left", on="product_id")
#        pairs = cudf.merge(pairs, self.item_cluster, how="left", on="product_id")
#        df = df.rename(columns={"cluster_id": "item_cluster_id"})
#        pairs = pairs.rename(columns={"cluster_id": "item_cluster_id"})
#
#        df = period_extraction(df, self.start_date, self.end_date)
#        df["day_diff"] = (self.end_date - df["time_stamp"]) / np.timedelta64(1, "D")
#        decay_rate = self.decay_rate
#        df["weight_decay"] = df["day_diff"].apply(lambda x: decay_rate**x)
#
#        action_list = [
#            ("cv", 3),
#            ("click", 2),
#            ("pv", 1),
#            ("other", 0),
#        ]
#        pairs["index"] = np.arange(len(pairs))
#        new_feature_names = []
#        for event_name, event_type in action_list:
#            feature_name = f"{event_name}-score-r{self.decay_rate}_by_itemgroup"
#            df["score"] = df["weight_decay"] * (df["event_type"] == event_type)
#            feature = df.groupby("item_cluster_id")["score"].sum().reset_index()
#            feature = feature.rename(columns={"score": feature_name})
#
#            pairs = cudf.merge(pairs, feature, how="left", on="item_cluster_id")
#            pairs[feature_name] = pairs[feature_name].fillna(0)
#            new_feature_names.append(feature_name)
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]
#
#
# class Item2ItemSimilarity(AbstractFeatureTransformer):
#    def __init__(
#        self,
#        start_date: pd.Timestamp,
#        end_date: pd.Timestamp,
#        model_dir: str,
#    ):
#        super().__init__(start_date, end_date)
#
#        self.category_types: List[str] = ["A", "B", "C", "D"]
#        self.item2vec = {}
#        for category in self.category_types:
#            with open(model_dir / f"item2vec_{category}.pickle", "rb") as f:
#                self.item2vec[category] = pickle.load(f)
#
#    def fit(self, df: cudf.DataFrame, pairs: cudf.DataFrame):
#        pass
#
#    def transform(self, df: cudf.DataFrame, pairs: cudf.DataFrame) -> cudf.DataFrame:
#        df = period_extraction(df, self.start_date, self.end_date)
#        pairs["index"] = np.arange(len(pairs))
#
#        # Remove other. (cv:3, click:2, view:1, other:0)
#        df = df[df["event_type"] > 0]
#
#        df = df.sort_values(["user_id", "time_stamp"], ascending=False)
#        df = df.groupby("user_id").agg({"product_id": list})
#
#        cand = pairs.groupby("user_id").agg({"product_id": list}).reset_index()
#
#        user_ids = []
#        candidate_items = []
#        avg_sim = []
#        max_sim = []
#        for user_id, cand_items in cand.to_pandas().values:
#            c = user_id.split("_")[1]
#            try:
#                items = df.loc[user_id, "product_id"][:10]
#            except:
#                items = cand_items
#            item_indices = [self.item2vec[c].wv.get_index(item) for item in items]
#            cand_item_indices = [
#                self.item2vec[c].wv.get_index(item) for item in cand_items
#            ]
#
#            V_items = self.item2vec[c].wv.vectors[item_indices]
#            V_cand_items = self.item2vec[c].wv.vectors[cand_item_indices]
#            item_norm = np.linalg.norm(V_items, axis=1, keepdims=True)
#            cand_norm = np.linalg.norm(V_cand_items, axis=1, keepdims=True)
#            item_norm[item_norm == 0] = 1
#            cand_norm[cand_norm == 0] = 1
#            V_items = V_items / item_norm
#            V_cand_items = V_cand_items / cand_norm
#
#            F = np.dot(V_cand_items, V_items.T)
#
#            user_ids += [user_id] * len(cand_items)
#            candidate_items += cand_items.tolist()
#            avg_sim += F.mean(axis=1).tolist()
#            max_sim += F.max(axis=1).tolist()
#
#        F = cudf.DataFrame(
#            {
#                "user_id": user_ids,
#                "product_id": candidate_items,
#                "avg_sim": avg_sim,
#                "max_sim": max_sim,
#            }
#        )
#        new_feature_names = [
#            "avg_sim",
#            "max_sim",
#        ]
#        pairs = cudf.merge(pairs, F, how="left", on=["user_id", "product_id"])
#        pairs = pairs.sort_values("index").drop("index", axis=1).reset_index(drop=True)
#        return pairs[new_feature_names]
#
