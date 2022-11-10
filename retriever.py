import argparse
import gc
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set

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
        "checkpoint_path",
        "dataset_name",
        "co_occur_items_path",
        "model_path",
    ]
    config = get_valid_path(config, check_param_list, date_th)
    return config


def get_retriever(config: Dict, date_th: str) -> ConcatRetriever:
    """Create a Retriever(candidate selector) according to Config.

    Args:
        config (Dict): Config.
        date_th (str): Boundary between train period and evaluation period.

    Returns:
        ConcatRetriever: Retriever(candidate selector).
    """

    retriever_config = prepare_config(config, date_th)

    # Define concat retriever.
    retrievers = [
        FavoriteItem(**retriever_config["favorite_item"]),
        CoOccurrenceItem(**retriever_config["co_occurence_item"]),
        RecBoleCF(**retriever_config["recbole_bpr"]),
        RecBoleCF(**retriever_config["recbole_itemknn"]),
        RecBoleCF(**retriever_config["recbole_recvae"]),
        Item2VecCF(**retriever_config["item2vec"]),
        PopularItem(**retriever_config["popular_item"]),
    ]
    concat_retriever_config = {"retrievers": retrievers, **config["concat_retriever"]}
    return ConcatRetriever(**concat_retriever_config)


def main(config_path):
    # Load Configuration.
    with open(config_path, "r") as f:
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
            retriever = get_retriever(deepcopy(config), date_th)
            retriever.fit(df)
            retriever.search(users)
            _ = retriever.evaluate(label, verbose=True)
            pairs = retriever.get_pairs(label)

        else:
            test = pd.read_csv(config["test_data_path"])
            users = test["user_id"].unique().tolist()
            retriever = get_retriever(deepcopy(config), date_th)
            retriever.fit(df)
            retriever.search(users)
            pairs = retriever.get_pairs(None)

        # Save pairs.
        pairs_path = pairs_dir / f"pairs_{date_th}_t{train_period}{suffix}.pickle"
        pairs[["user_id", "product_id", "target"]].to_pickle(pairs_path)
        pairs_rank_path = (
            pairs_dir / f"pairs_rank_{date_th}_t{train_period}{suffix}.pickle"
        )
        pairs.to_pickle(pairs_rank_path)

        del df, pairs, retriever
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/retriever.yaml")
    args = parser.parse_args()
    main(args.config_path)
