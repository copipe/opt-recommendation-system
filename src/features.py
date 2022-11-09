from __future__ import annotations

import gc
import pickle
from abc import ABCMeta, abstractmethod
from calendar import c
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from gensim.models import word2vec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, get_trainer, init_seed
from tqdm import tqdm

from src.utils import flatten_2d_list, order_immutable_deduplication


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


class RecentActionDayDiff(AbstractFeatureTransformer):
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
            action_list = [
                ("cv", 3),
                ("click", 2),
                ("pv", 1),
                ("other", 0),
            ]

            groupby_keys = [
                # "user_id",
                # "product_id",
                ["user_id", "product_id"],
            ]

            end_date = df["time_stamp"].max()
            df["day_diff"] = (end_date - df["time_stamp"]) / np.timedelta64(1, "D")

            new_feature_names = []
            for gkey in groupby_keys:
                str_gkey = "-".join(gkey) if isinstance(gkey, list) else gkey
                for event_name, event_type in action_list:
                    feature_name = f"{event_name}-daydiff_by_{str_gkey}"
                    feature = (
                        df[df["event_type"] == event_type]
                        .groupby(gkey)["day_diff"]
                        .min()
                        .reset_index()
                    )
                    feature = feature.rename(columns={"day_diff": feature_name})
                    pairs = pd.merge(pairs, feature, how="left", on=gkey)
                    pairs[feature_name] = pairs[feature_name].fillna(999)
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
        # configurations initialization
        if not self.checkpoint_path.exists():
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
                train_data,
                valid_data,
                saved=True,
                show_progress=config["show_progress"],
            )

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.checkpoint_path.exists():
                self.fit()

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


class RecentActionItems(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
        n_items: int = 5,
    ):
        super().__init__(save_file_path)
        self.n_items = n_items

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            # Remove other. (cv:3, click:2, view:1, other:0)
            df = df[df["event_type"] > 0]

            df = df.sort_values(["user_id", "time_stamp"], ascending=False)
            df = df.groupby("user_id").agg({"product_id": list})
            df = df.reset_index()

            feature = []
            for user_id, items in tqdm(df.values):
                items = order_immutable_deduplication(items)
                items = items[: self.n_items]
                if len(items) < self.n_items:
                    items = items + ["[PAD]"] * (self.n_items - len(items))
                feature.append([user_id] + items)
            new_feature_names = [f"item_before{i+1}" for i in range(self.n_items)]
            feature = pd.DataFrame(feature, columns=["user_id"] + new_feature_names)
            pairs = pd.merge(pairs, feature, how="left", on="user_id")
            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class RecentActionItemsByAction(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
        n_items: int = 5,
    ):
        super().__init__(save_file_path)
        self.n_items = n_items

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

            new_feature_names = []
            for event_name, event_type in action_list:
                df_ = df[df["event_type"] == event_type]
                df_ = df_.sort_values(["user_id", "time_stamp"], ascending=False)
                df_ = df_.groupby("user_id").agg({"product_id": list})
                df_ = df_.reset_index()

                feature = []
                for user_id, items in tqdm(df_.values):
                    items = order_immutable_deduplication(items)
                    items = items[: self.n_items]
                    if len(items) < self.n_items:
                        items = items + ["[PAD]"] * (self.n_items - len(items))
                    feature.append([user_id] + items)
                feature_names = [
                    f"item_before{i+1}_{event_name}" for i in range(self.n_items)
                ]
                feature = pd.DataFrame(feature, columns=["user_id"] + feature_names)
                pairs = pd.merge(pairs, feature, how="left", on="user_id")
                pairs = pairs.fillna("[PAD]")
                new_feature_names += feature_names
            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class RecBoleItem2ItemSim(AbstractFeatureTransformer):
    def __init__(
        self,
        recent_action_items_config: dict,
        save_file_path: str,
        model_name: str,
        dataset_name: str,
        config_file_list: List[Path],
        checkpoint_path: str | Path | None = None,
        batch_size: int = 16,
    ):
        super().__init__(save_file_path)
        self.recent_action_items_config = recent_action_items_config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file_list = config_file_list
        if isinstance(checkpoint_path, str):
            self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.name = f"{self.name}-{model_name}"

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        # configurations initialization
        if not self.checkpoint_path.exists():
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
                train_data,
                valid_data,
                saved=True,
                show_progress=config["show_progress"],
            )

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.checkpoint_path.exists():
                self.fit()

            # load RecentActionItems feature
            rai = RecentActionItems(**self.recent_action_items_config)
            recent_action_items = rai.fit_transform(df, pairs)
            recent_action_items_cols = recent_action_items.columns.tolist()
            n_items = len(recent_action_items_cols)

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

            item_token2id = dataset.field2token_id["item_id"]

            data = {}
            data["product_id"] = torch.LongTensor(
                list(map(lambda pid: item_token2id.get(pid, 0), pairs["product_id"]))
            )
            for col in recent_action_items_cols:
                data[col] = torch.LongTensor(
                    list(
                        map(
                            lambda pid: item_token2id.get(pid, 0),
                            recent_action_items[col],
                        )
                    )
                )
            data = Interaction(data)

            bs = self.batch_size
            n_iterations = len(pairs) // bs + 1
            pred = {c: [] for c in recent_action_items_cols}
            for i in tqdm(range(n_iterations)):
                batch_data = data[i * bs : (i + 1) * bs]
                batch_data = batch_data.to("cuda")
                for col in recent_action_items_cols:
                    # model inference
                    batch_pred = model.item_similarity(batch_data, "product_id", col)
                    pred[col] += batch_pred.to("cpu").detach().tolist()
            pred = pd.DataFrame(pred)

            new_feature_names = []
            feat_name = f"max_{self.model_name}-sim_n{n_items}_item2history"
            pairs[feat_name] = pred.max(axis=1)
            new_feature_names.append(feat_name)

            feat_name = f"avg_{self.model_name}-sim_n{n_items}_item2history"
            pairs[feat_name] = pred.mean(axis=1)
            new_feature_names.append(feat_name)

            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class RecBoleItem2ItemSimByAction(AbstractFeatureTransformer):
    def __init__(
        self,
        recent_action_items_by_action_config: dict,
        save_file_path: str,
        model_name: str,
        dataset_name: str,
        config_file_list: List[Path],
        checkpoint_path: str | Path | None = None,
        batch_size: int = 16,
    ):
        super().__init__(save_file_path)
        self.recent_action_items_by_action_config = recent_action_items_by_action_config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file_list = config_file_list
        if isinstance(checkpoint_path, str):
            self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.name = f"{self.name}-{model_name}"

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        # configurations initialization
        if not self.checkpoint_path.exists():
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
                train_data,
                valid_data,
                saved=True,
                show_progress=config["show_progress"],
            )

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.checkpoint_path.exists():
                self.fit()

            # load RecentActionItemsByAction feature
            rai = RecentActionItemsByAction(**self.recent_action_items_by_action_config)
            recent_action_items = rai.fit_transform(df, pairs)
            recent_action_items_cols = recent_action_items.columns.tolist()

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

            item_token2id = dataset.field2token_id["item_id"]

            data = {}
            data["product_id"] = torch.LongTensor(
                list(map(lambda pid: item_token2id.get(pid, 0), pairs["product_id"]))
            )
            for col in recent_action_items_cols:
                data[col] = torch.LongTensor(
                    list(
                        map(
                            lambda pid: item_token2id.get(pid, 0),
                            recent_action_items[col],
                        )
                    )
                )
            data = Interaction(data)

            bs = self.batch_size
            n_iterations = len(pairs) // bs + 1
            pred = {c: [] for c in recent_action_items_cols}
            for i in tqdm(range(n_iterations)):
                batch_data = data[i * bs : (i + 1) * bs]
                batch_data = batch_data.to("cuda")
                for col in recent_action_items_cols:
                    # model inference
                    batch_pred = model.item_similarity(batch_data, "product_id", col)
                    pred[col] += batch_pred.to("cpu").detach().tolist()
            pred = pd.DataFrame(pred)

            cols_by_action = {a: [] for a in ["cv", "click", "pv", "other"]}
            for col in recent_action_items_cols:
                event_name = col.split("_")[-1]
                cols_by_action[event_name].append(col)

            new_feature_names = []
            for event_name, cols in cols_by_action.items():
                n_items = len(cols)
                feat_name = (
                    f"max_{self.model_name}-sim_n{n_items}-{event_name}_item2history"
                )
                pairs[feat_name] = pred[cols].max(axis=1)
                new_feature_names.append(feat_name)

                feat_name = (
                    f"avg_{self.model_name}-sim_n{n_items}-{event_name}_item2history"
                )
                pairs[feat_name] = pred[cols].mean(axis=1)
                new_feature_names.append(feat_name)

                pred = pred.drop(cols, axis=1)

            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class Item2VecItem2ItemSim(AbstractFeatureTransformer):
    def __init__(
        self,
        recent_action_items_config: dict,
        save_file_path: str,
        model_path: Path,
        vector_size: int = 128,
        window: int = 5,
        epochs: int = 10,
        ns_exponent: float = 0.75,
        min_count: int = 5,
        seed: int = 0,
        workers: int = 8,
        batch_size=256,
    ):
        super().__init__(save_file_path)
        self.recent_action_items_config = recent_action_items_config
        self.model_path = Path(model_path)
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.ns_exponent = ns_exponent
        self.seed = seed
        self.min_count = min_count
        self.workers = workers
        self.batch_size = batch_size

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        if not self.model_path.exists():
            df = df.sort_values(["user_id", "time_stamp"], ascending=False)
            df = df.groupby("user_id").agg({"product_id": list})
            df = df.reset_index()

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

            with open(self.model_path, "wb") as f:
                result = {
                    "model": item2vec,
                    "similar_items": {},
                }
                pickle.dump(result, f)

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.model_path.exists():
                self.fit(df, pairs)

            # load RecentActionItems feature
            rai = RecentActionItems(**self.recent_action_items_config)
            recent_action_items = rai.fit_transform(df, pairs)
            recent_action_items_cols = recent_action_items.columns.tolist()
            n_items = len(recent_action_items_cols)

            # load model
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)["model"]

            data = {}
            data["product_id"] = pairs["product_id"].tolist()
            for col in recent_action_items_cols:
                data[col] = recent_action_items[col].tolist()

            bs = self.batch_size
            n_iterations = len(pairs) // bs + 1
            pred = {c: [] for c in recent_action_items_cols}
            for i in tqdm(range(n_iterations)):
                for col in recent_action_items_cols:
                    # model inference
                    items1 = data["product_id"][i * bs : (i + 1) * bs]
                    items2 = data[col][i * bs : (i + 1) * bs]
                    batch_pred = self._get_item_similarity(model, items1, items2)
                    pred[col] += batch_pred

            pred = pd.DataFrame(pred)
            new_feature_names = []
            feat_name = f"max_item2vec-sim_n{n_items}_item2history"
            pairs[feat_name] = pred.max(axis=1)
            new_feature_names.append(feat_name)

            feat_name = f"avg_item2vec-sim_n{n_items}_item2history"
            pairs[feat_name] = pred.mean(axis=1)
            new_feature_names.append(feat_name)

            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature

    def _get_item_similarity(self, model, items1, items2):
        item2vec_vocab = set(model.wv.index_to_key)
        items1_e = np.zeros((len(items1), self.vector_size))
        items2_e = np.zeros((len(items2), self.vector_size))

        for i, (item1, item2) in enumerate(zip(items1, items2)):
            if item1 in item2vec_vocab:
                items1_e[i, :] = model.wv.get_vector(item1)
            if item2 in item2vec_vocab:
                items2_e[i, :] = model.wv.get_vector(item2)
        items1_e = self._normalize(items1_e)
        items2_e = self._normalize(items2_e)
        return (items1_e * items2_e).sum(axis=1).tolist()

    def _normalize(self, vectors):
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vectors / norm


class Item2VecItem2ItemSimByAction(AbstractFeatureTransformer):
    def __init__(
        self,
        recent_action_items_by_action_config: dict,
        save_file_path: str,
        model_path: Path,
        vector_size: int = 128,
        window: int = 5,
        epochs: int = 10,
        ns_exponent: float = 0.75,
        min_count: int = 5,
        seed: int = 0,
        workers: int = 8,
        batch_size=256,
    ):
        super().__init__(save_file_path)
        self.recent_action_items_by_action_config = recent_action_items_by_action_config
        self.model_path = Path(model_path)
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.ns_exponent = ns_exponent
        self.seed = seed
        self.min_count = min_count
        self.workers = workers
        self.batch_size = batch_size

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        if not self.model_path.exists():
            df = df.sort_values(["user_id", "time_stamp"], ascending=False)
            df = df.groupby("user_id").agg({"product_id": list})
            df = df.reset_index()

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

            with open(self.model_path, "wb") as f:
                result = {
                    "model": item2vec,
                    "similar_items": {},
                }
                pickle.dump(result, f)

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.model_path.exists():
                self.fit(df, pairs)

            # load RecentActionItemsByAction feature
            rai = RecentActionItemsByAction(**self.recent_action_items_by_action_config)
            recent_action_items = rai.fit_transform(df, pairs)
            recent_action_items_cols = recent_action_items.columns.tolist()

            # load model
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)["model"]

            data = {}
            data["product_id"] = pairs["product_id"].tolist()
            for col in recent_action_items_cols:
                data[col] = recent_action_items[col].tolist()

            bs = self.batch_size
            n_iterations = len(pairs) // bs + 1
            pred = {c: [] for c in recent_action_items_cols}
            for i in tqdm(range(n_iterations)):
                for col in recent_action_items_cols:
                    # model inference
                    items1 = data["product_id"][i * bs : (i + 1) * bs]
                    items2 = data[col][i * bs : (i + 1) * bs]
                    batch_pred = self._get_item_similarity(model, items1, items2)
                    pred[col] += batch_pred
            pred = pd.DataFrame(pred)
            new_feature_names = []

            cols_by_action = {a: [] for a in ["cv", "click", "pv", "other"]}
            for col in recent_action_items_cols:
                event_name = col.split("_")[-1]
                cols_by_action[event_name].append(col)

            for event_name, cols in cols_by_action.items():
                n_items = len(cols)
                feat_name = f"max_item2vec-sim_n{n_items}-{event_name}_item2history"
                pairs[feat_name] = pred[cols].max(axis=1)
                new_feature_names.append(feat_name)

                feat_name = f"avg_item2vec-sim_n{n_items}-{event_name}_item2history"
                pairs[feat_name] = pred[cols].mean(axis=1)
                new_feature_names.append(feat_name)

                pred = pred.drop(cols, axis=1)

            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature

    def _get_item_similarity(self, model, items1, items2):
        item2vec_vocab = set(model.wv.index_to_key)
        items1_e = np.zeros((len(items1), self.vector_size))
        items2_e = np.zeros((len(items2), self.vector_size))

        for i, (item1, item2) in enumerate(zip(items1, items2)):
            if item1 in item2vec_vocab:
                items1_e[i, :] = model.wv.get_vector(item1)
            if item2 in item2vec_vocab:
                items2_e[i, :] = model.wv.get_vector(item2)
        items1_e = self._normalize(items1_e)
        items2_e = self._normalize(items2_e)
        return (items1_e * items2_e).sum(axis=1).tolist()

    def _normalize(self, vectors):
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vectors / norm


class CoOccurItems(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
        co_occur_items_path: str,
        n_items: int = 5,
    ):
        super().__init__(save_file_path)
        self.n_items = n_items
        self.co_occur_items_path = Path(co_occur_items_path)

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            if not self.co_occur_items_path.exists():
                raise "co_occur_items_path does not exist."

            with open(self.co_occur_items_path, "rb") as f:
                co_occur_items = pickle.load(f)

            feature = []
            for item, co_items in tqdm(co_occur_items.items()):
                co_items = co_items[: self.n_items]
                if len(co_items) < self.n_items:
                    co_items = co_items + ["[PAD]"] * (self.n_items - len(co_items))
                feature.append([item] + co_items)
            new_feature_names = [f"item_co-occur{i+1}" for i in range(self.n_items)]
            feature = pd.DataFrame(feature, columns=["product_id"] + new_feature_names)
            pairs = pd.merge(pairs, feature, how="left", on="product_id")
            feature = pairs[new_feature_names]
            self._save_feature(feature)
            return feature


class RecentActionSimItems(AbstractFeatureTransformer):
    def __init__(
        self,
        save_file_path: str,
        decay_rates: List,
        co_occur_items_config: Dict,
    ):
        super().__init__(save_file_path)
        self.decay_rates = decay_rates
        self.co_occur_items_config = co_occur_items_config

    def fit(self, df: pd.DataFrame, pairs: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.save_file_path.exists():
            return self._load_feature()
        else:
            # load CoOccurItems feature
            coi = CoOccurItems(**self.co_occur_items_config)
            co_occur_items = coi.fit_transform(df, pairs)
            co_occur_items_cols = co_occur_items.columns.tolist()
            co_occur_items = pd.concat([co_occur_items, pairs[["user_id"]]], axis=1)
            n_items = len(co_occur_items_cols)

            decay_rates = self.decay_rates
            end_date = df["time_stamp"].max()
            df["day_diff"] = (end_date - df["time_stamp"]) / np.timedelta64(1, "D")
            for i, decay_rate in enumerate(decay_rates):
                df[f"weight_decay{i}"] = df["day_diff"].apply(lambda x: decay_rate**x)
                df[f"score{i}"] = (df["event_type"] + 1) * df[f"weight_decay{i}"]
            score_cols = [f"score{i}" for i in range(len(decay_rates))]
            df = df.groupby(["user_id", "product_id"])[score_cols].sum().reset_index()

            for col in co_occur_items_cols:
                co_occur_items = (
                    pd.merge(
                        co_occur_items,
                        df,
                        how="left",
                        left_on=["user_id", col],
                        right_on=["user_id", "product_id"],
                    )
                    .drop(["product_id", col], axis=1)
                    .fillna(0)
                )
                col_map = {sc: f"{sc}_by_{col}" for sc in score_cols}
                co_occur_items = co_occur_items.rename(columns=col_map)

            new_feature_names = []

            for i in range(len(decay_rates)):
                new_cols = [f"score{i}_by_{col}" for col in co_occur_items_cols]
                feat_name = f"avg_simitems-score_n{n_items}_r{decay_rates[i]}"
                co_occur_items[feat_name] = co_occur_items[new_cols].mean(axis=1)
                new_feature_names.append(feat_name)
                feat_name = f"max_simitems-score_n{n_items}_r{decay_rates[i]}"
                co_occur_items[feat_name] = co_occur_items[new_cols].max(axis=1)
                new_feature_names.append(feat_name)

            feature = co_occur_items[new_feature_names]
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
            print(f"{feature_transformer.name} fiting...")
            feature_transformer.fit(df, pairs)

    def transform(self, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        features = []
        for feature_transformer in self.feature_transformers:
            print(f"{feature_transformer.name} transforming...")
            features.append(feature_transformer.transform(df, pairs))
            gc.collect()
            torch.cuda.empty_cache()
        return pd.concat(features, axis=1)
