import datasets as ds
import kagglehub, logging

def load_dataset():
    logging.basicConfig(
        level=logging.INFO, 
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    dataset = "tobiasbueck/multilingual-customer-support-tickets"
    subset = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    # Download latest version
    kagglehub.dataset_download(dataset)

    # Load a DataFrame with a specific version of a CSV
    df: ds.Dataset = kagglehub.dataset_load(
        adapter = kagglehub.KaggleDatasetAdapter.HUGGING_FACE,
        handle = dataset,
        path = subset
    )
    queue_labels = df.unique("queue")
    # df = df.to_iterable_dataset()

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
    id2label = {i: label for i, label in enumerate(queue_labels)}
    label2id = {label: i for i, label in enumerate(queue_labels)}
    df_en = df_en.map(lambda x: {
        "queue_encoded" : label2id[x.get("queue")]
    })
    df_en = df_en.class_encode_column("queue_encoded")
    df_en = df_en.select_columns(["ticket", "queue_encoded"]).rename_columns({"ticket": "text", "queue_encoded": "labels"})
    logging.info(df_en.to_pandas())
    df_en = df_en.train_test_split(test_size=0.25, train_size=0.75, stratify_by_column="labels", seed=42)
    return df_en, queue_labels, id2label, label2id