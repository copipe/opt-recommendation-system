from typing import Dict, List

import numpy as np
import pandas as pd

from src.metrics import ndcg_score
from src.utils import get_data_period, period_extraction


def get_pred_items(
    df: pd.DataFrame, y_pred: np.ndarray, top_n: int = 22
) -> Dict[str, List[str]]:
    """Select recommended items for each user.

    Args:
        df (pd.DataFrame): Dataframe with user_id and product_id columns.
        y_pred (np.ndarray): Predicted values of the model for each user_id and product_id (higher values should be recommended).
        top_n (int): Maximum number of recommended items. Defaults to 22.

    Returns:
        Dict[str, List[str]]: Recommended items for each user.
    """
    assert len(df) == len(y_pred), "df and y_pred must have equal length."
    df = df.copy()
    df["y_pred"] = y_pred

    df = df.sort_values(["user_id", "y_pred"], ascending=False).reset_index(drop=True)
    pred_items_df = df.groupby(["user_id"]).agg({"product_id": list}).reset_index()

    pred_items = {}
    for user_id, items in pred_items_df.values:
        pred_items[user_id] = items[:top_n]
    return pred_items


def get_ndcg_score(
    df: pd.DataFrame,
    pred_items: Dict[str, List[str]],
    date_th: str,
    train_period: int,
    eval_period: int = 7,
    ndcg_k: int = 22,
) -> float:
    """_summary_

    Args:
        df (pd.DataFrame): Preprocessed data.
        pred_items (Dict[str, List[str]]): Recommended items for each user.
        date_th (str): Boundary between train period and evaluation period.
        train_period (int): Number of days of train period.
        eval_period (int): Number of days of evaluation period. Defaults to 7.
        ndcg_k (int): Maximum number of recommended items. Defaults to 22.

    Returns:
        float: NDCG@K
    """

    # Extract only the data for the evaluation period
    _, _, eval_start, eval_end = get_data_period(date_th, train_period, eval_period)
    df = period_extraction(df, eval_start, eval_end)

    # Remove other and (cv=1 and ad !=1). (cv:3, click:2, view:1, other:0)
    df = df[df["event_type"] > 0]
    df = df[(df["event_type"] != 3) | (df["ad"] == 1)]

    # LB's rated users are probably filtered.
    users = df[df["event_type"] > 1]["user_id"].unique()
    df = df[df["user_id"].isin(users)]

    # Extract positive examples for each user.
    df = (
        df.groupby(["user_id", "product_id"])[["event_type", "time_stamp"]]
        .max()
        .reset_index()
    )
    df = df.sort_values(["user_id", "event_type"], ascending=False)
    df = df.reset_index()
    df = (
        df.groupby(["user_id"])
        .agg({"event_type": list, "product_id": list})
        .reset_index()
    )

    # Calculate evaluation metrics. (NDCG@K)
    n_users = 0
    sum_ndcg = 0
    for user_id, true_scores, true_items in df.values:
        true_score_mapper = {k: v for k, v in zip(true_items, true_scores)}
        pred_items_ = pred_items.get(user_id, [])
        pred_scores = [true_score_mapper.get(item, 0) for item in pred_items_]

        n_users += 1
        sum_ndcg += ndcg_score(true_scores, pred_scores, ndcg_k)
    ndcg = sum_ndcg / n_users
    return ndcg
