from pathlib import Path

import cudf

TRAIN_DIR = Path("data/raw/train/")
OUTPUT_PATH = "data/processed/train.csv"


def main() -> None:
    """4種（A, B, C, D)のtrain.csvを結合(concat)して保存"""
    train_by_category = []
    for train_path in TRAIN_DIR.iterdir():
        train = cudf.read_csv(train_path, delimiter="\t")
        train["category"] = train["user_id"].str.split("_", expand=True)[1]
        train_by_category.append(train)

    train = cudf.concat(train_by_category).reset_index(drop=True)
    train.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
