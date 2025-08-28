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
    df_en = df_en.select(["subject", "body", "queue"])
    df_en = df_en.fill_null("")
    return df_en

def process_dataset(model, dataset, batch_size: int = 32, chunk_size: int = 500):
    preds = []
    for idx_chunk, df_chunk in enumerate(dataset.iter_slices(n_rows=chunk_size)):
        for idx_batch, df_batch in enumerate(df_chunk.iter_slices(n_rows=batch_size)):
          print(f"Processing chunk {idx_chunk}: Batch {idx_batch}...")
          chunk_preds = model.process_batch(tickets=df_batch["ticket"].to_list())
          preds.extend(chunk_preds)

    return dataset.with_columns(pl.Series("pred_queue_category", preds))