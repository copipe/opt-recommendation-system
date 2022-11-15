import argparse
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation import get_ndcg_score, get_pred_items
from src.ranker import CATRanker, LGBRanker


def prepare_data(df: pd.DataFrame, config: Dict, train: bool):
    for col in config["category_cols"]:
        df[col] = df[col].astype("category")

    if train and config["negative_sampling"]:
        frac = config["negative_sampling_rate"]
        random_state = config["negative_sampling_random_state"]
        df = pd.concat(
            [
                df[df["target"] > 0],
                df[df["target"] == 0].sample(frac=frac, random_state=random_state),
            ]
        ).reset_index(drop=True)

    df = df.sort_values("user_id").reset_index(drop=True)
    X = df.drop(config["drop_cols"], axis=1)
    y = df[config["target_col"]]
    if config["model_type"] == "lgb":
        q = df.groupby("user_id").size().values
        return df, X, y, q
    else:
        g = df["user_id"]
        return df, X, y, g


def save_feature_importance(model, output_dir):
    feature_importance = (
        pd.DataFrame(
            {
                "feature": model.feature_names_,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(50)
    )
    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(feature_importance)), feature_importance["importance"])
    plt.xticks(
        np.arange(len(feature_importance)), feature_importance["feature"], rotation=-90
    )
    plt.savefig(output_dir / "feature_importance.png", bbox_inches="tight")
    plt.close()


def main(config_path):
    # Load config.
    with open(config_path, "r") as f:
        config_org = yaml.safe_load(f)

    n_repeat = config_org["negative_sampling_n_repeat"]
    for i in range(n_repeat):
        # Set random state.
        config = deepcopy(config_org)
        config["negative_sampling_random_state"] = i

        # Load data.
        df = pd.read_pickle(config["preprocessed_train_path"])
        train = pd.read_pickle(config["train_features_path"])
        valid = pd.read_pickle(config["valid_features_path"])
        test = pd.read_pickle(config["test_features_path"])

        # Prepare model inputs.
        train, X_train, y_train, z_train = prepare_data(train, config, True)
        valid, X_valid, y_valid, z_valid = prepare_data(valid, config, False)
        test, X_test, _, _ = prepare_data(test, config, False)

        # Train & Inference
        if config["model_type"] == "lgb":
            model = LGBRanker()
        else:
            model = CATRanker()

        model.train(
            config["model_params"],
            X_train,
            y_train,
            z_train,
            X_valid,
            y_valid,
            z_valid,
            train_params=config["train_params"],
        )

        if i == 0:
            y_valid_pred = model.predict(X_valid)
            y_test_pred = model.predict(X_test)
        else:
            y_valid_pred += model.predict(X_valid)
            y_test_pred += model.predict(X_test)
    y_valid_pred = y_valid_pred / n_repeat
    y_test_pred = y_test_pred / n_repeat

    # Culculate CV score.
    pred_items = get_pred_items(valid[["user_id", "product_id"]], y_valid_pred)
    score = get_ndcg_score(
        df,
        pred_items,
        config["cv_params"]["date_th"],
        config["cv_params"]["train_period"],
        config["cv_params"]["eval_period"],
    )
    print(f"ndcg cv score: {score:.4f}")

    # Make submission file.
    pred_items = get_pred_items(test[["user_id", "product_id"]], y_test_pred)
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

    # Save feature importance.
    if config["model_type"] == "lgb":
        save_feature_importance(model, output_dir)

    # Save prediction for valid data.
    valid_pred = valid[["user_id", "product_id"]].assign(y_pred=y_valid_pred)
    valid_pred.to_pickle(output_dir / "valid_pred.pickle")

    # Save prediction for test data.
    test_pred = test[["user_id", "product_id"]].assign(y_pred=y_test_pred)
    test_pred.to_pickle(output_dir / "test_pred.pickle")

    # Save CV score.
    score_df = pd.DataFrame({"ndcg": [score]})
    score_df.to_csv(output_dir / "score.csv", index=False)

    # Save submission file.
    filename = f"{config['model_name']}_cv-{score:.4f}_submission.tsv"
    submission.to_csv(output_dir / filename, sep="\t", index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/ranker.yaml")
    args = parser.parse_args()
    main(args.config_path)
