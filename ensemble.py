import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from src.evaluation import get_ndcg_score, get_pred_items


def weight_averaging(config:Dict, filename:str) -> pd.DataFrame:
    # Load setting.
    model_path_list = config["model_path_list"]
    model_weight_list = np.array(config["model_weight_list"])
    model_weights = model_weight_list / model_weight_list.sum()
    n_models = len(model_path_list)
    pred_cols = [f"y_pred{i}" for i in range(n_models)]

    # Merge predictions.
    for i in range(n_models):
        pred_path = Path(model_path_list[i], filename)
        tmp_pred = pd.read_pickle(pred_path)
        if i == 0:
            pred = tmp_pred.rename(columns={"y_pred": pred_cols[i]})
        else:
            pred = pd.merge(
                pred,
                tmp_pred.rename(columns={"y_pred": pred_cols[i]}),
                how="left",
                on=["user_id", "product_id"],
            )

    # Convert score to rank.
    if config["method"] == "rank":
        for col in pred_cols:
            pred[col] = pred.groupby("user_id")[col].rank()

    # Weight averaging.
    for col, weight in zip(pred_cols, model_weights):
        pred[col] = pred[col] * weight
    pred["y_pred"] = pred[pred_cols].mean(axis=1)
    return pred


def main(config_path):
    # Load config.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data.
    df = pd.read_pickle(config["preprocessed_train_path"])
    valid_pred = weight_averaging(config, 'valid_pred.pickle')
    test_pred = weight_averaging(config, 'test_pred.pickle')

    # Culculate CV score.
    pred_items = get_pred_items(valid_pred[["user_id", "product_id"]], valid_pred["y_pred"])
    score = get_ndcg_score(
        df,
        pred_items,
        config["cv_params"]["date_th"],
        config["cv_params"]["train_period"],
        config["cv_params"]["eval_period"],
    )
    print(f"ndcg cv score: {score:.4f}")

    # Make submission file.
    pred_items = get_pred_items(test_pred[["user_id", "product_id"]], test_pred["y_pred"])
    submission = []
    for user, items in pred_items.items():
        for i, item in enumerate(items):
            submission.append([user, item, i])
    submission = pd.DataFrame(submission)

    # Make output_dir.
    output_dir = Path(config["save_file_dir"], config["model_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config file.
    with open(output_dir / "config.pickle", "wb") as f:
        pickle.dump(config, f)


    # Save prediction for valid data.
    valid_pred = valid_pred[['user_id', 'product_id', 'y_pred']]
    valid_pred.to_pickle(output_dir / "valid_pred.pickle")

    # Save prediction for test data.
    test_pred = test_pred[['user_id', 'product_id', 'y_pred']]
    test_pred.to_pickle(output_dir / "test_pred.pickle")

    # Save CV score.
    score_df = pd.DataFrame({"ndcg": [score]})
    score_df.to_csv(output_dir / "score.csv", index=False)

    # Save submission file.
    filename = f"{config['model_name']}_cv-{score:.4f}_submission.tsv"
    submission.to_csv(output_dir / filename, sep="\t", index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/ensemble.yaml")
    args = parser.parse_args()
    main(args.config_path)
