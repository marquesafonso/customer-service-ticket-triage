import datasets as ds
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_dataset():
    dataset = "tobiasbueck/multilingual-customer-support-tickets"
    subset = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    # Download latest version
    kagglehub.dataset_download(dataset)

    # Load a DataFrame with a specific version of a CSV
    df: ds.Dataset = kagglehub.dataset_load(
        adapter = KaggleDatasetAdapter.HUGGING_FACE,
        handle = dataset,
        path = subset
    )
    queue_labels = df.unique("queue")
    # df = df.to_iterable_dataset()

    df.to_iterable_dataset()
    df_en = df.filter(lambda x: x["language"] == "en")
    df_en = df_en.select_columns(["subject", "body", "queue"])
    df_en = df_en.map(lambda x: {
        "subject": x.get("subject", "") or "",
        "body": x.get("body", "") or "",
        "queue": x.get("queue")
    })
    df_en = df_en.map(lambda x: {
        "ticket": x.get("subject") + "\n" + x.get("body")
        })
    return df_en, queue_labels