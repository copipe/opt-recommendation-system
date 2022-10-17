from typing import List, Tuple

import cudf
import pandas as pd


def order_immutable_deduplication(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving array order.
    (Assumed to be applied to the candidate item list)

    Args:
        items (List[str]): Item list.

    Returns:
        _type_: Deduped item list.
    """
    return sorted(set(items), key=items.index)


def flatten_2d_list(list_2d: List[List[str]]) -> List[str]:
    """Convert 2d list to 1d.

    Args:
        list_2d (List[List[str]]): 2d list before conversion.

    Returns:
        List[str]: 1d list after conversion.
    """
    return sum(list_2d, [])


def get_data_period(
    date_th: str, train_period: int, eval_period: int
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Calculate the first and last days of the train period and evaluation period.

    Args:
        date_th (str): Boundary between train period and evaluation period.
        train_period (int): Number of days of train period.
        eval_period (int, optional): Number of days of evaluation period. Defaults to 7.

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]: First and last day of study period and evaluation period.
    """
    train_start = pd.to_datetime(date_th) - pd.Timedelta(train_period, "days")
    train_end = pd.to_datetime(date_th)
    eval_start = pd.to_datetime(date_th)
    eval_end = pd.to_datetime(date_th) + pd.Timedelta(eval_period, "days")
    return train_start, train_end, eval_start, eval_end


def period_extraction(
    df: cudf.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> cudf.DataFrame:
    """Extract data for specific period.

    Args:
        df (cudf.DataFrame): Preprocessed data.
        start_date (pd.Timestamp): First day of extraction period.
        end_date (pd.Timestamp): Last day of extraction period.

    Returns:
        cudf.DataFrame: Data after extraction.
    """
    df = df[
        (start_date < df["time_stamp"]) & (df["time_stamp"] <= end_date)
    ].reset_index(drop=True)
    return df
