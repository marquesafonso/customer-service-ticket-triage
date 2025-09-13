# Customer Service Ticket Triage

This project is based on the kaggle dataset [Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/) which contains labeled email tickets with agents answer, priorities, queues.

The dataset subset used was: aa_dataset-tickets-multi-lang-5-2-50-version.csv

Our focus here is on categorizing the emails with tickets (i.e., subject + body) into the relevant queue for further processing. We filter tickets to have only english ones.

The methodology followed has been to take a pretrained model [Deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small) and finetune it to our data for single class classification.

We also experimented with [Deberta v3 zeroshot 2.0](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0)

<details>
  <summary><b>Quickstart and dependency overview</b></summary>
  
  ### Quickstart
  If you would like to reproduce this, simply follow this steps. **Assumes python 3.13 and uv are installed.**

  ```
  git clone https://github.com/marquesafonso/customer-service-ticket-triage
  cd customer-service-ticket-triage
  ```
  Add an **.env** file following what is provided in **.env.example** - place there your huggingface access token (this will be relevant if you'd like to push your model to the hub). Then:
  ```
  uv sync
  uv run main.py
  ```
  And your model will train.

  ### Dependency overview
  Please note that this code is optimized for intel gpu usage with pytorch. Using CUDA should be simpler, but here I will outline where you might change the code to adapt to the device of your choosing.

  To install pytorch with xpu support I followed [this](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch) guide.

  Here are the project dependencies:

  ```
    [project]
    name = "customer-service-ticket-triage"
    version = "0.1.0"
    description = "Add your description here"
    readme = "README.md"
    requires-python = ">=3.13"
    dependencies = [
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "pytorch-triton-xpu>=3.3.0 ; sys_platform == 'win32' or sys_platform == 'linux'",
        "accelerate>=1.10.1",
        "hf-xet>=1.1.9",
        "huggingface-hub>=0.34.4",
        "kagglehub[hf-datasets]>=0.3.13",
        "python-dotenv>=1.1.1",
        "trackio>=0.3.2",
        "transformers>=4.55.4",
        "evaluate>=0.4.5",
        "scikit-learn>=1.7.2",
        "sentencepiece>=0.2.1",
        "protobuf>=6.32.1",
    ]

    [tool.uv.sources]
    torch = [
    { index = "pytorch-xpu", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
    ]
    torchvision = [
    { index = "pytorch-xpu", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
    ]
    pytorch-triton-xpu = [
    { index = "pytorch-xpu", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
    ]

    [[tool.uv.index]]
    name = "pytorch-xpu"
    url = "https://download.pytorch.org/whl/xpu"
    explicit = true
  ```

  To change the device you will need to adapt the installation of the packages accordingly and then change the following lines in *src/models.py*

  ```
  class TicketTriageModel:
    def __init__(
        self,
        labels: list,
        model_name = "microsoft/deberta-v3-small",
        id2label: dict[int, Any] = None
    ):
        (...)
        self.model_name = model_name
        if torch.xpu.is_available():
            self.device = "xpu"
            torch.xpu.empty_cache()
        else:
            self.device = "cpu"
  ```

  ```
  class BaseModel:
    def __init__(
        self,
        labels: list,
        model_name = "microsoft/deberta-v3-small"
    ):
        (...)
        def finetune_model(
            self,
            train_dataset : ds.Dataset,
            validation_dataset: ds.Dataset,
            id2label : dict,
            label2id : dict,
            batch_size : int = 8,
            num_train_epochs : int = 5
    ):
        logging.info(f"XPU: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
  ```
</details>


## Preparing the dataset

First we download the dataset from kaggle using the hf-datasets adapter as we'll use the datasets library later for finetuning the model.

```{python}

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

```

Preprocessing the dataset. We fill null values and merge subject and body into a single "ticket" column. The queue values are encoded into class labels and label2id and id2label mappings are produced. Further, we rename columns according to the transformers conventions (i.e., "text" and "labels").

```
queue_labels = df.unique("queue")


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
```

Then, we split the dataset into train (75%) and test (25%), and further split the training set into train (90% of 75%) and validation (10% of 75%) sets. We also get the feel of the class imbalance problem that we have at hands by checking the value counts of each label. Both splits are done with a stratify_by_column argument set to "labels" such that we preserve the distributions of the queue classes among splits.

```
## Splitting dataset into train, validation and test sets
train_valid, test = df_en.train_test_split(test_size=0.25, stratify_by_column="labels", seed=42).values()
train, valid = train_valid.train_test_split(test_size=0.1, stratify_by_column="labels", seed=42).values()

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
```

## Preparing the base model

Now we prepare a base model class with attributes and methods that will be used for training dataset.

You will note that the **preprocess_data** method tokenizes the batches of training examples using the tokenizer of the pretrained model.

The **compute_metrics** will calculate the f1 with macro averaging to deal with class imbalance (see [this guide](https://www.numberanalytics.com/blog/f1-score-imbalanced-classes-guide)), accuracy and per-class f1 scores to allow for proper monitoring of the training procedure which we will to [Trackio](https://huggingface.co/blog/trackio), as you will see later on.

Take into consideration that we are optimising for the f1-macro metric and not accuracy as we'd like for the model to have good recall capabilities for underrepresented classes (as opposed to a learning strategy geared towards precision only.)

```
class BaseModel:
    def __init__(
        self,
        labels: list,
        model_name = "microsoft/deberta-v3-small"
    ):
        logging.basicConfig(
            level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        warnings.filterwarnings("ignore")
        self.model_name = model_name
        self.labels = labels
        self.f1_metric = evaluate.load("f1")
        self.accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(self, eval_preds: EvalPrediction):
        logits, labels = eval_preds
        preds = np.argmax(logits, -1)
        ## using f1-macro to address class imbalance
        ## See https://www.numberanalytics.com/blog/f1-score-imbalanced-classes-guide
        macro_f1 = self.f1_metric.compute(predictions=preds, references=labels, average="macro")

        ## Accuracy
        acc = self.accuracy_metric.compute(predictions=preds, references=labels)
        
        ## Per-class f1 score
        per_class_f1 = self.f1_metric.compute(predictions=preds, references=labels, average=None)
        per_class_f1_dict = {f"f1_class_{i}": score for i, score in enumerate(per_class_f1["f1"])}

        return {
            "accuracy": acc["accuracy"],
            "f1_macro": macro_f1["f1"],
            **per_class_f1_dict
        }
    
    def preprocess_data(self, dataset: ds.Dataset):
        ## Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        def tokenize(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset = dataset.with_format("torch")
        return dataset
```

Now comes the denser part - the model finetuning method - which I will break down.

We start by loading the pretrained model and provide the number of labels, id2label and label2id mappings.

```
def finetune_model(
        self,
        train_dataset : ds.Dataset,
        validation_dataset: ds.Dataset,
        id2label : dict,
        label2id : dict,
        batch_size : int = 8,
        num_train_epochs : int = 5,
        oversample : bool = False
    ):
        logging.info(f"XPU: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        
        ## Load model
        logging.info("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            problem_type="single_label_classification", # problem_type (str, optional) — Problem type for XxxForSequenceClassification models. Can be one of "regression", "single_label_classification" or "multi_label_classification".
            num_labels=len(self.labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
```

Optionally we may oversample underrepresented classes. This relates to an empirical observation during training that individual f1-scores of specific classes were 0.0 even at the end of training. See more about oversampling [here](https://medium.com/@abdallahashraf90x/oversampling-for-better-machine-learning-with-imbalanced-data-68f9b5ac2696). The default behaviour is not to oversample.

```
if oversample:
    ## Oversampling (underrepresented) classes
    train_dataset = oversample_with_interleave(
        train_dataset,
        num_labels=len(self.labels),
        seed=10
    )
```

<details>
  <summary><b>See the oversampling implementation</b></summary>
  
  The implementation for oversampling is available in *src/dataset.py*. It uses the *datasets* library interleave_datasets method to oversample classes according to their probabilities. We can then use the boost_classes and boost_factor arguments to increase the probabilities of specific classes. This is a regularization technique and is optional, you may enable it by setting oversample = True in the finetune_model method of BaseModel, this defaults to a balanced dataset with no class boosting..

  ```
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
  ```
</details>

Then, we tokenize and prepare the training dataset for training
```
self.encoded_train_dataset = self.preprocess_data(train_balanced)
self.encoded_validation_dataset = self.preprocess_data(validation_dataset)
logging.info(self.encoded_train_dataset)
```

Before we prepare the Trainer, we initiliaze the training arguments. These will be crucial in how the model is finetuned. Some of the key elements here:

- Logging metrics (of compute_metrics) using trackio for training monitoring.
- Regularization techniques used to prevent overfitting (Graddient clipping and Gradient accumulation, Weight decay, learning rate scheduler)
- Objective metric is f1-macro

```
## Initialize training args
args = TrainingArguments(
    f"{HF_USER}/ticket_triage_{self.model_name.replace('/','_')}_finetuned",
    save_strategy = "epoch",
    eval_strategy="epoch",
    num_train_epochs=num_train_epochs,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2, ## Efective batch size will be batch_size * gradient_accumulation_steps
    weight_decay=0.01,
    max_grad_norm=0.5,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    report_to="trackio",
    remove_unused_columns=False,
    push_to_hub=True
)
```

With these arguments, we can provide the encoded training and validation datasets to the trainer, the tokenizer, as well as the training arguments and the metrics to be computed for logging. You will notice that an Early Stopping callback is used with a patient parameter of 2 (i.e., the model will wait 2 eval epochs/steps of deteriorating objective function and then exit training. In our case this means 2 consecutive epochs of non-increasing f1-macro). The goal is to use resources as efficiently as possible

```
## Initialize the trainer
self.trainer = Trainer(
    self.model,
    args,
    train_dataset=self.encoded_train_dataset,
    eval_dataset=self.encoded_validation_dataset,
    tokenizer=self.tokenizer,
    compute_metrics=self.compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
```
With everything ready, we get the train to start training.

```
# fine tune the model
self.trainer.train()
```

## Putting it all together

With all these elements prepared you are ready to finetune a base model according to your specified arguments, monitor its training, then initialize a TicketTriageModel instance with the new model (this is a model class for inference - zero shot classification) and evaluate its accuracy on the test dataset.

This can be achieved by simply running:

```
uv run main.py
```

One of the key elements of this approach is that it is easy to reproduce in different machines (do not forget to change the device of BaseModel if not xpu or cpu), potentially allowing one to run it on a server with better hardware resources.

Below you will find the contents of *main.py*. 

```
def main():
    logging.basicConfig(
        level=logging.INFO, 
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # log into HF
    logging.info("Logging into HF...")
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER =  os.getenv("HF_USER")
    login(token=HF_TOKEN)

    ## Prepare datasets
    logging.info("Prepare datasets...")
    dataset, queue_labels, id2label, label2id = load_dataset()
    train_dataset, validation_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

    ## Prepare base model
    logging.info("Preparing base model...")
    basemodel_name = "microsoft/deberta-v3-small"
    basemodel = BaseModel(
        model_name=basemodel_name,
        labels=queue_labels
    )

    ## finetune the model
    logging.info("Finetuning the model...")
    basemodel.finetune_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        id2label=id2label,
        label2id=label2id,
        batch_size=8,
        num_train_epochs=7
    )

    ## Evaluate on test set
    logging.info("Evaluating on test set...")
    finetuned_model_name = f"{HF_USER}/ticket_triage_{basemodel_name.replace('/',"_")}_finetuned"
    finetuned_model = TicketTriageModel(
        model_name=finetuned_model_name,
        labels=queue_labels,
        id2label=id2label
    )

    finetuned_dataset = finetuned_model.get_predictions_from_dataset(test_dataset, batch_size=64)
    finetuned_dataset.to_parquet(f"output/{basemodel_name.replace('/',"_")}_finetuned_preds.parquet")
    
    # From a saved predictions file
    # import datasets as ds
    # finetuned_dataset = ds.Dataset.from_parquet(f"output/{basemodel_name.replace('/',"_")}_finetuned_preds.parquet")

    logging.info(finetuned_dataset.to_pandas().head())

    finetuned_accuracy = finetuned_dataset.filter(lambda x: x["pred_labels"] == x["labels_str"]).num_rows * 100 / finetuned_dataset.num_rows
    logging.info(finetuned_accuracy)


if __name__ == "__main__":
    main()
```

## Results and Experiments

TODO

## Citations

1. https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/
2. https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch
3. https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer#transformers.TrainingArguments
4. https://medium.com/@_prinsh_u/gradient-clipping-accumulation-and-more-essential-techniques-for-effective-training-c08f59c8b15d
5. https://medium.com/@abdallahashraf90x/oversampling-for-better-machine-learning-with-imbalanced-data-68f9b5ac2696
6. https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
7. https://www.numberanalytics.com/blog/f1-score-imbalanced-classes-guide
8. https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
9. https://huggingface.co/blog/trackio
10. https://huggingface.co/docs/transformers/main/en/training
11. https://github.com/huggingface/evaluate/
12. https://github.com/MoritzLaurer/zeroshot-classifier/
13. https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0