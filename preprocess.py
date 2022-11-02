from pathlib import Path

import pandas as pd
import yaml

from src.utils import get_data_period, period_extraction

CONFIG_PATH = "config/preprocess.yaml"


def concat_train_dataframes(train_dir: Path) -> pd.DataFrame:
    """Concat train data (splited by 4 categories).

    Args:
        train_dir (Path): Train data directory.

    Returns:
        pd.DataFrame: Train data with 4files combined.
    """
    train_dataframes = []
    for data_path in train_dir.iterdir():
        train = pd.read_csv(data_path, delimiter="\t")
        train["category"] = train["user_id"].str.split("_", expand=True)[1]
        train["time_stamp"] = pd.to_datetime(train["time_stamp"])
        train_dataframes.append(train)
    return pd.concat(train_dataframes).reset_index(drop=True)


def recbole_formatter(df: pd.DataFrame) -> pd.DataFrame:
    """Convert train data to recbole format.

    Args:
        df (pd.DataFrame): Train data.

    Returns:
        pd.DataFrame: Formated train data.
    """
    usecols = ["user_id", "product_id", "event_type", "time_stamp"]
    for col in usecols:
        assert col in df.columns, f"Input dataframe(df) must have '{col}' column."
    df = df[usecols].copy()

    df["time_stamp"] = df["time_stamp"].apply(lambda t: t.timestamp())
    df = df.rename(
        columns={
            "user_id": "user_id:token",
            "product_id": "item_id:token",
            "event_type": "rating:float",
            "time_stamp": "timestamp:float",
        }
    )
    return df


def label_formatter(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> pd.DataFrame:
    """Create a label table for evaluation.

    Args:
        df_train (pd.DataFrame): Data for the train period. Used for user filtering..
        df_eval (pd.DataFrame): Data for the evaluation period.

    Returns:
        pd.DataFrame: Label data containing columns of "user_id", "event_type", "product_id"
    """
    # Remove other and (cv=1 and ad !=1). (cv:3, click:2, view:1, other:0)
    df_eval = df_eval[df_eval["event_type"] > 0]
    df_eval = df_eval[(df_eval["event_type"] != 3) | (df_eval["ad"] == 1)]

    # LB's rated users are probably filtered.
    eval_users = df_eval[df_eval["event_type"] > 1]["user_id"].unique()
    df_eval = df_eval[df_eval["user_id"].isin(eval_users)]

    # Extract only users who have logs during the train period.
    train_users = df_train["user_id"].unique()
    df_eval = df_eval[df_eval["user_id"].isin(train_users)]

    # Extract positive examples for each user.
    df_eval = (
        df_eval.groupby(["user_id", "product_id"])[["event_type", "time_stamp"]]
        .max()
        .reset_index()
    )
    df_eval = df_eval.sort_values(["user_id", "event_type"], ascending=False)
    df_eval = df_eval.groupby(["user_id"]).agg({"event_type": list, "product_id": list})
    df_eval = df_eval.reset_index()

    return df_eval


if __name__ == "__main__":

    # Load Configuration.
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    train_period = config["train_period"]
    eval_period = config["eval_period"]
    output_dir = Path(config["processed_data_dir"])
    train_dir = Path(config["input_data_dir"]) / "train"

    # Save concatenated train data.
    df = concat_train_dataframes(train_dir)
    df.to_pickle(output_dir / "train.pickle")

    for date_th in config["date_th"]:
        # Get data period.
        data_period = get_data_period(date_th, train_period, eval_period)
        train_start, train_end, eval_start, eval_end = data_period

        # Split data.
        df_train = period_extraction(df, train_start, train_end)
        df_eval = period_extraction(df, eval_start, eval_end)

        # Save train data.
        df_train.to_pickle(output_dir / f"train_{date_th}_t{train_period}.pickle")

        # Save recbole formatted train data.
        df_train_recbole = recbole_formatter(df_train)
        recbole_dir = output_dir / f"recbole_{date_th}_t{train_period}"
        recbole_path = recbole_dir / f"recbole_{date_th}_t{train_period}.inter"
        recbole_dir.mkdir(exist_ok=True, parents=True)
        df_train_recbole.to_csv(recbole_path, index=False, sep="\t")

        # Save label data.
        df_label = label_formatter(df_train, df_eval)
        df_label.to_pickle(output_dir / f"label_{date_th}.pickle")
