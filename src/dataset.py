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

    ## Splitting dataset into train, validation and test sets
    train_valid, test = df_en.train_test_split(test_size=0.25, stratify_by_column="labels", seed=10).values()
    train, valid = train_valid.train_test_split(test_size=0.1, stratify_by_column="labels", seed=10).values()

    ## Verifying distribution of class labels in train and validation datasets
    labels = sorted(train.to_pandas()["labels"].unique())
    for l in labels:
        logging.info(f"[Train] Label {l}: {train.to_pandas()["labels"].apply(lambda x: x == l).sum()} occurrences")

    labels = sorted(valid.to_pandas()["labels"].unique())
    for l in labels:
        logging.info(f"[Validation] Label {l}: {valid.to_pandas()["labels"].apply(lambda x: x == l).sum()} occurrences")

    dataset_dict = ds.DatasetDict({
        "train": train,
        "validation": valid,
        "test": test
    })

    return dataset_dict, queue_labels, id2label, label2id


def oversample_with_interleave(train_dataset, num_labels, boost_classes=None, boost_factor=2.0, seed=42):
    """
    Oversample minority classes using interleave_datasets.
    
    Args:
        train_dataset: datasets.Dataset (with "labels" column)
        num_labels: number of unique labels
        boost_classes: list of class indices to oversample more strongly
        boost_factor: multiplier for boost_classes probabilities
        seed: random seed
    """
    ## Split into one dataset per class
    class_datasets = []
    class_sizes = []
    for c in range(num_labels):
        ds_c = train_dataset.filter(lambda x: x["labels"] == c)
        class_datasets.append(ds_c)
        class_sizes.append(len(ds_c))

    ## Base probabilities: proportional to dataset sizes
    total = sum(class_sizes)
    probs = [size / total for size in class_sizes]

    ## Optionally boost certain classes (e.g. underrepresented)
    if boost_classes is not None:
        for c in boost_classes:
            probs[c] *= boost_factor

    ## Normalize to sum to 1
    s = sum(probs)
    probs = [p / s for p in probs]

    logging.info("Sampling probabilities:", {c: round(p, 3) for c, p in enumerate(probs)})

    ## Interleave with oversampling
    interleaved = ds.interleave_datasets(
        class_datasets,
        probabilities=probs,
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    return interleaved