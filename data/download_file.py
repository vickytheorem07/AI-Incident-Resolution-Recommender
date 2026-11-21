from pathlib import Path

import pandas as pd


DATASET_URI = (
    "hf://datasets/s2pidape/support-ticket-dataset/with_ai/"
    "train-00000-of-00001.parquet"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset.csv"


def main():
    df = pd.read_parquet(DATASET_URI)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
