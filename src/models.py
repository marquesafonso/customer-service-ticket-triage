import warnings, logging
import datasets as ds
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
import torch
import numpy as np
import evaluate

class TicketTriageModel:
    def __init__(
        self,
        labels: list,
        model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    ):
        warnings.filterwarnings("ignore")
        logging.basicConfig(
           level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.model_name = model_name
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        self.classifier = pipeline("zero-shot-classification", model=self.model_name, device=self.device)
        self.labels = labels

    def get_predictions_from_dataset(self, dataset: ds.Dataset, batch_size: int = 32) -> ds.Dataset:
        """
        Run batch inference on a Hugging Face Dataset and add predictions as a column.
        """

        def predict(batch):
            outputs = self.classifier(batch["ticket"], self.labels,
                                      multi_label=False, batch_size=batch_size)
            if isinstance(outputs, dict):
                return {"pred_queue": outputs["labels"][0]}
            else:
                return {"pred_queue": [out["labels"][0] for out in outputs]}

        return dataset.map(predict, batched=True, batch_size=batch_size)

class BaseModel:
    def __init__(
        self,
        labels: list,
        model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    ):
        logging.basicConfig(
            level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        warnings.filterwarnings("ignore")
        self.model_name = model_name
        self.labels = labels

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def compute_metrics(self, eval_preds: EvalPrediction):
        metric = evaluate.load("f1")
        logits, labels = eval_preds
        preds = np.argmax(logits, -1)
        return metric.compute(predictions=preds, references=labels)
    
    def preprocess_data(self, dataset: ds.Dataset):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        def tokenize(examples):
            return self.tokenizer(examples, padding="max_length", truncation=True)
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.with_format("torch")
        return dataset
            
    def finetune_model(
        self,
        train_dataset : ds.Dataset,
        id2label : dict,
        label2id : dict,
        batch_size : int = 16
    ):
        logging.info(f"XPU: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        
        # load model
        logging.info("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            problem_type="zero-shot-classification", 
            num_labels=len(self.labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        # tokenize and prepare the training dataset for training
        logging.info("Tokenizing and prepare the training dataset for training...")
        self.encoded_train_dataset = self.preprocess_data(train_dataset)
        logging.info(self.encoded_train_dataset)

        # initialize training args
        args = TrainingArguments(
            f"marquesafonso/ticket_triage_{self.model_name.replace('/','_')}_finetuned",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            fp16=True,
            lr_scheduler_type="cosine",
            report_to="trackio",
            remove_unused_columns=False,
            push_to_hub=True
        )

        # initialize the trainer
        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=self.encoded_train_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        # fine tune the model
        self.trainer.train()

