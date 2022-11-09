import gc
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml

from src.features import (
    ConcatFeatureTransformer,
    CoOccurItems,
    Item2VecItem2ItemSim,
    Item2VecItem2ItemSimByAction,
    ItemAttribute,
    RecBoleItem2ItemSim,
    RecBoleItem2ItemSimByAction,
    RecBolePredictor,
    RecentActionDayDiff,
    RecentActionFrequency,
    RecentActionItemsByAction,
    RecentActionSimItems,
)

CONFIG_PATH = "config/features.yaml"


def get_features_config(config: Dict, date_th: str) -> Dict:
    raf1_config = config["recent_action_frequency1"].copy()
    raf1_config["save_file_path"] = raf1_config["save_file_path"][date_th]

    raf2_config = config["recent_action_frequency2"].copy()
    raf2_config["save_file_path"] = raf2_config["save_file_path"][date_th]

    itemattr_config = config["item_attribute"].copy()
    itemattr_config["save_file_path"] = itemattr_config["save_file_path"][date_th]

    radd_config = config["recent_action_daydiff"].copy()
    radd_config["save_file_path"] = radd_config["save_file_path"][date_th]

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

    ## Prepare config of recbole-neumf retriever.
    # train_period = config["train_period"]
    # neumf_config = config["recbole_neumf"].copy()
    # neumf_config["checkpoint_path"] = neumf_config["checkpoint_path"][date_th]
    # neumf_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    # neumf_config["save_file_path"] = neumf_config["save_file_path"][date_th]

    rai_config = config["recent_action_items"].copy()
    rai_config["save_file_path"] = rai_config["save_file_path"][date_th]

    train_period = config["train_period"]
    bpr_itemsim_config = config["recbole_item2item_sim_bpr"].copy()
    bpr_itemsim_config["checkpoint_path"] = bpr_itemsim_config["checkpoint_path"][
        date_th
    ]
    bpr_itemsim_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    bpr_itemsim_config["save_file_path"] = bpr_itemsim_config["save_file_path"][date_th]
    bpr_itemsim_config["recent_action_items_config"] = rai_config

    # train_period = config["train_period"]
    # neumf_itemsim_config = config["recbole_item2item_sim_neumf"].copy()
    # neumf_itemsim_config["checkpoint_path"] = neumf_itemsim_config["checkpoint_path"][
    #    date_th
    # ]
    # neumf_itemsim_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    # neumf_itemsim_config["save_file_path"] = neumf_itemsim_config["save_file_path"][
    #    date_th
    # ]
    # neumf_itemsim_config["recent_action_items_config"] = rai_config

    train_period = config["train_period"]
    item2vec_itemsim_config = config["item2vec_item2item_sim"].copy()
    item2vec_itemsim_config["model_path"] = item2vec_itemsim_config["model_path"][
        date_th
    ]
    item2vec_itemsim_config["save_file_path"] = item2vec_itemsim_config[
        "save_file_path"
    ][date_th]
    item2vec_itemsim_config["recent_action_items_config"] = rai_config

    coi1_config = config["co_occur_items1"].copy()
    coi1_config["save_file_path"] = coi1_config["save_file_path"][date_th]
    coi1_config["co_occur_items_path"] = coi1_config["co_occur_items_path"][date_th]

    # coi2_config = config["co_occur_items2"].copy()
    # coi2_config["save_file_path"] = coi2_config["save_file_path"][date_th]
    # coi2_config["co_occur_items_path"] = coi2_config["co_occur_items_path"][date_th]

    rasi1_config = config["recent_action_sim_items1"].copy()
    rasi1_config["save_file_path"] = rasi1_config["save_file_path"][date_th]
    rasi1_config["co_occur_items_config"] = coi1_config

    # rasi2_config = config["recent_action_sim_items2"].copy()
    # rasi2_config["save_file_path"] = rasi2_config["save_file_path"][date_th]
    # rasi2_config["co_occur_items_config"] = coi2_config

    raiba_config = config["recent_action_items_by_action"].copy()
    raiba_config["save_file_path"] = raiba_config["save_file_path"][date_th]

    train_period = config["train_period"]
    bpr_itemsim_ba_config = config["recbole_item2item_sim_bpr_by_action"].copy()
    bpr_itemsim_ba_config["checkpoint_path"] = bpr_itemsim_ba_config["checkpoint_path"][
        date_th
    ]
    bpr_itemsim_ba_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"
    bpr_itemsim_ba_config["save_file_path"] = bpr_itemsim_ba_config["save_file_path"][
        date_th
    ]
    bpr_itemsim_ba_config["recent_action_items_by_action_config"] = raiba_config

    train_period = config["train_period"]
    item2vec_itemsim_ba_config = config["item2vec_item2item_sim_by_action"].copy()
    item2vec_itemsim_ba_config["model_path"] = item2vec_itemsim_ba_config["model_path"][
        date_th
    ]
    item2vec_itemsim_ba_config["save_file_path"] = item2vec_itemsim_ba_config[
        "save_file_path"
    ][date_th]
    item2vec_itemsim_ba_config["recent_action_items_by_action_config"] = raiba_config

    features_config = {
        "recent_action_frequency1": raf1_config,
        "recent_action_frequency2": raf2_config,
        "item_attribute": itemattr_config,
        "recent_action_daydiff": radd_config,
        "recbole_bpr": bpr_config,
        "recbole_itemknn": itemknn_config,
        "recbole_recvae": recvae_config,
        # "recbole_neumf": neumf_config,
        "recent_action_items": rai_config,
        "recbole_item2item_sim_bpr": bpr_itemsim_config,
        # "recbole_item2item_sim_neumf": neumf_itemsim_config,
        "item2vec_item2item_sim": item2vec_itemsim_config,
        "recent_action_sim_items1": rasi1_config,
        # "recent_action_sim_items2": rasi2_config,
        "recent_action_items_by_action": raiba_config,
        "recbole_item2item_sim_bpr_by_action": bpr_itemsim_ba_config,
        "recent_action_items_by_action": item2vec_itemsim_ba_config,
    }
    return features_config


def get_feature_transformer(config: Dict, date_th: str) -> ConcatFeatureTransformer:

    feature_config = get_features_config(config, date_th)

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
        RecentActionSimItems(**feature_config["recent_action_sim_items1"]),
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
