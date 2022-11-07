import gc
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml

from src.features import (
    ConcatFeatureTransformer,
    ItemAttribute,
    RecBolePredictor,
    RecentActionFrequency,
)

CONFIG_PATH = "config/features.yaml"


def get_features_config(config: Dict, date_th: str) -> Dict:
    raf1_config = config["recent_action_frequency1"].copy()
    raf1_config["save_file_path"] = raf1_config["save_file_path"][date_th]

    raf2_config = config["recent_action_frequency2"].copy()
    raf2_config["save_file_path"] = raf2_config["save_file_path"][date_th]

    itemattr_config = config["item_attribute"].copy()
    itemattr_config["save_file_path"] = itemattr_config["save_file_path"][date_th]

    # Prepare config of recbole-bpr retriever.
    train_period = config["train_period"]
    bpr_config = config["recbole_bpr"].copy()
    bpr_config["checkpoint_path"] = bpr_config["checkpoint_path"][date_th]
    bpr_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    bpr_config["save_file_path"] = bpr_config["save_file_path"][date_th]

    # Prepare config of recbole-itemknn retriever.
    train_period = config["train_period"]
    itemknn_config = config["recbole_itemknn"].copy()
    itemknn_config["checkpoint_path"] = itemknn_config["checkpoint_path"][date_th]
    itemknn_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    itemknn_config["save_file_path"] = itemknn_config["save_file_path"][date_th]

    # Prepare config of recbole-recvae retriever.
    train_period = config["train_period"]
    recvae_config = config["recbole_recvae"].copy()
    recvae_config["checkpoint_path"] = recvae_config["checkpoint_path"][date_th]
    recvae_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    recvae_config["save_file_path"] = recvae_config["save_file_path"][date_th]

    features_config = {
        "recent_action_frequency1": raf1_config,
        "recent_action_frequency2": raf2_config,
        "item_attribute": itemattr_config,
        "recbole_bpr": bpr_config,
        "recbole_itemknn": itemknn_config,
        "recbole_recvae": recvae_config,
    }
    return features_config


def get_feature_transformer(config: Dict, date_th: str) -> ConcatFeatureTransformer:

    feature_config = get_features_config(config, date_th)

    feature_transformers = [
        ItemAttribute(**feature_config["item_attribute"]),
        RecentActionFrequency(**feature_config["recent_action_frequency1"]),
        RecentActionFrequency(**feature_config["recent_action_frequency2"]),
        RecBolePredictor(**feature_config["recbole_bpr"]),
        RecBolePredictor(**feature_config["recbole_itemknn"]),
        RecBolePredictor(**feature_config["recbole_recvae"]),
    ]
    return ConcatFeatureTransformer(feature_transformers)


if __name__ == "__main__":

    # Load Configuration.
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    input_dir = Path(config["preprocessed_data_dir"])
    pairs_dir = Path(config["pairs_dir"])
    features_dir = Path(config["features_dir"])
    pairs_suffix = (
        f"_{config['pairs_file_suffix']}" if config["pairs_file_suffix"] else ""
    )
    features_suffix = (
        f"_{config['features_file_suffix']}" if config["features_file_suffix"] else ""
    )
    train_period = config["train_period"]

    for date_th in config["date_th"]:

        df = pd.read_pickle(input_dir / f"train_{date_th}_t{train_period}.pickle")
        pairs = pd.read_pickle(
            pairs_dir / f"pairs_{date_th}_t{train_period}{pairs_suffix}.pickle"
        )

        feature_transformer = get_feature_transformer(config, date_th)
        features = feature_transformer.fit_transform(df, pairs)
        pairs = pairs[["user_id", "product_id", "target"]]
        features = pd.concat([pairs, features], axis=1)
        features.to_pickle(
            features_dir / f"features_{date_th}_t{train_period}{features_suffix}.csv",
        )

        del df, pairs, feature_transformer, features
        gc.collect()
        torch.cuda.empty_cache()
