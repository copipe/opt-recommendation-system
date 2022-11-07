import gc
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml

from src.retriever import (
    ConcatRetriever,
    CoOccurrenceItem,
    FavoriteItem,
    Item2VecCF,
    PopularItem,
    RecBoleCF,
)

CONFIG_PATH = "config/retriever.yaml"


def get_retriever_config(config: Dict, date_th: str) -> Dict[str, Dict]:
    # Prepare config of co-occurence retriever.
    train_period = config["train_period"]
    co_occur_items_dir = Path(config["co_occur_items_dir"])
    co_occur_items_path = (
        co_occur_items_dir / f"co-occurrence_{date_th}_t{train_period}.pickle"
    )
    co_occurence_config = {
        "co_occur_items_path": co_occur_items_path,
        **config["co_occurence_item"],
    }

    # Prepare config of item2vec-cf retriever.
    item2vec_config = config["item2vec"].copy()
    item2vec_config["model_path"] = item2vec_config["model_path"][date_th]

    # Prepare config of recbole-bpr retriever.
    bpr_config = config["recbole_bpr"].copy()
    bpr_config["checkpoint_path"] = bpr_config["checkpoint_path"][date_th]
    bpr_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"

    # Prepare config of recbole-itemknn retriever.
    itemknn_config = config["recbole_itemknn"].copy()
    itemknn_config["checkpoint_path"] = itemknn_config["checkpoint_path"][date_th]
    itemknn_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"

    # Prepare config of recbole-recvae retriever.
    recvae_config = config["recbole_recvae"].copy()
    recvae_config["checkpoint_path"] = recvae_config["checkpoint_path"][date_th]
    recvae_config["dataset_name"] = f"recbole_{date_th}_t{train_period}"

    retriever_config = {
        "popular_item": config["popular_item"],
        "favorite_item": config["favorite_item"],
        "co_occurence_item": co_occurence_config,
        "item2vec_item": item2vec_config,
        "recbole_bpr": bpr_config,
        "recbole_itemknn": itemknn_config,
        "recbole_recvae": recvae_config,
    }
    return retriever_config


def get_retriever(config: Dict, date_th: str) -> ConcatRetriever:
    """Create a Retriever(candidate selector) according to Config.

    Args:
        config (Dict): Config.
        date_th (str): Boundary between train period and evaluation period.

    Returns:
        ConcatRetriever: Retriever(candidate selector).
    """

    retriever_config = get_retriever_config(config, date_th)

    # Define concat retriever.
    retrievers = [
        FavoriteItem(**retriever_config["favorite_item"]),
        CoOccurrenceItem(**retriever_config["co_occurence_item"]),
        RecBoleCF(**retriever_config["recbole_bpr"]),
        RecBoleCF(**retriever_config["recbole_itemknn"]),
        RecBoleCF(**retriever_config["recbole_recvae"]),
        Item2VecCF(**retriever_config["item2vec_item"]),
        PopularItem(**retriever_config["popular_item"]),
    ]
    concat_retriever_config = {"retrievers": retrievers, **config["concat_retriever"]}
    return ConcatRetriever(**concat_retriever_config)


if __name__ == "__main__":

    # Load Configuration.
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    input_dir = Path(config["preprocessed_data_dir"])
    pairs_dir = Path(config["pairs_dir"])
    suffix = f"_{config['pairs_file_suffix']}" if config["pairs_file_suffix"] else ""
    train_period = config["train_period"]

    for date_th in config["date_th"]:

        df = pd.read_pickle(input_dir / f"train_{date_th}_t{train_period}.pickle")

        if date_th != "2017-04-30":
            label = pd.read_pickle(input_dir / f"label_{date_th}.pickle")
            users = label["user_id"].unique().tolist()
            retriever = get_retriever(config, date_th)
            retriever.fit(df)
            retriever.search(users)
            scores = retriever.evaluate(label, verbose=True)
            pairs = retriever.get_pairs(label)

        else:
            test = pd.read_csv(config["test_data_path"])
            users = test["user_id"].unique().tolist()
            retriever = get_retriever(config, date_th)
            retriever.fit(df)
            retriever.search(users)
            pairs = retriever.get_pairs(None)

        # Save pairs.
        pairs_path = pairs_dir / f"pairs_{date_th}_t{train_period}{suffix}.pickle"
        pairs.to_pickle(pairs_path)

        del df, pairs, retriever
        torch.cuda.empty_cache()
        gc.collect()
