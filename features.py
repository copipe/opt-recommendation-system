import argparse
import gc
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
import yaml

from src.features import (
    ConcatFeatureTransformer,
    Item2VecItem2ItemSim,
    ItemAttribute,
    RecBoleItem2ItemSim,
    RecBolePredictor,
    RecentActionDayDiff,
    RecentActionFrequency,
    RecentActionSimItems,
    RetrieverRank,
)


def get_valid_path(
    param: Dict, check_param_list: List, date_th: str, visited: Set = set()
):
    for key, value in param.items():
        if key in check_param_list:
            param[key] = param[key][date_th]
        elif isinstance(value, dict) and key not in visited:
            visited = visited | set([key])
            param[key] = get_valid_path(param[key], check_param_list, date_th, visited)
    return param


def prepare_config(config: Dict, date_th: str) -> Dict:
    check_param_list = [
        "save_file_path",
        "checkpoint_path",
        "dataset_name",
        "co_occur_items_path",
        "model_path",
        "pairs_rank_path",
    ]
    config = get_valid_path(config, check_param_list, date_th)
    return config


def get_feature_transformer(config: Dict, date_th: str) -> ConcatFeatureTransformer:

    feature_config = prepare_config(config, date_th)

    feature_transformers = [
        ItemAttribute(**feature_config["item_attribute"]),
        RecentActionFrequency(**feature_config["recent_action_frequency1"]),
        RecentActionFrequency(**feature_config["recent_action_frequency2"]),
        RecentActionDayDiff(**feature_config["recent_action_daydiff"]),
        RecBolePredictor(**feature_config["recbole_bpr"]),
        RecBolePredictor(**feature_config["recbole_itemknn"]),
        RecBolePredictor(**feature_config["recbole_recvae"]),
        RecBoleItem2ItemSim(**feature_config["recbole_item2item_sim_bpr"]),
        Item2VecItem2ItemSim(**feature_config["item2vec_item2item_sim"]),
        RecentActionSimItems(**feature_config["recent_action_sim_items"]),
        RetrieverRank(**feature_config["retriever_rank"]),
    ]
    return ConcatFeatureTransformer(feature_transformers)


def main(config_path):
    # Load Configuration.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    input_dir = Path(config["preprocessed_data_dir"])
    pairs_dir = Path(config["pairs_dir"])
    features_dir = Path(config["features_dir"])
    pairs_suffix = "_" + config["pairs_file_suffix"]
    features_suffix = "_" + config["features_file_suffix"]
    train_period = config["train_period"]

    for date_th in config["date_th"]:
        df_path = input_dir / f"train_{date_th}_t{train_period}.pickle"
        pairs_path = pairs_dir / f"pairs_{date_th}_t{train_period}{pairs_suffix}.pickle"
        features_path = (
            features_dir / f"features_{date_th}_t{train_period}{features_suffix}.pickle"
        )

        # make features.
        df = pd.read_pickle(df_path)
        pairs = pd.read_pickle(pairs_path)
        feature_transformer = get_feature_transformer(deepcopy(config), date_th)
        features = feature_transformer.fit_transform(df, pairs)
        pairs = pairs[["user_id", "product_id", "target"]]
        features = pd.concat([pairs, features], axis=1)
        features.to_pickle(features_path)

        del df, pairs, feature_transformer, features
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/features.yaml")
    args = parser.parse_args()
    main(args.config_path)
