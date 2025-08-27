import polars as pl
import kagglehub
from kagglehub import KaggleDatasetAdapter, PolarsFrameType

def load_dataset():
    dataset = "tobiasbueck/multilingual-customer-support-tickets"
    subset = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    # Download latest version
    kagglehub.dataset_download(dataset)

    # Load a DataFrame with a specific version of a CSV
    df: pl.DataFrame = kagglehub.dataset_load(
        adapter = KaggleDatasetAdapter.POLARS,
        handle = dataset,
        path = subset,
        polars_frame_type=PolarsFrameType.DATA_FRAME
    )

    df_en = df.filter(pl.col("language") == "en")
    return df_en