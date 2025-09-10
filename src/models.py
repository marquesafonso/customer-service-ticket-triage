import warnings, logging
from typing import Any
import datasets as ds
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    EvalPrediction
)
import torch
import numpy as np
import evaluate

class TicketTriageModel:
    def __init__(
        self,
        labels: list,
        model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        id2label: dict[int, Any] = None
    ):
        warnings.filterwarnings("ignore")
        logging.basicConfig(
           level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.model_name = model_name
        if torch.xpu.is_available():
            self.device = "xpu"
            torch.xpu.empty_cache()
        else:
            self.device = "cpu"
        self.classifier = pipeline("zero-shot-classification", model=self.model_name, device=self.device)
        self.labels = labels
        if not id2label:
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.ids2labels = self.config.id2label
        else:
            self.ids2labels = id2label

    def get_predictions_from_dataset(self, dataset: ds.Dataset, batch_size: int = 32) -> ds.Dataset:
        """
        Run batch inference on a Hugging Face Dataset and add predictions as a column.
        """
        def ids2labels(batch):
            return {"labels_str" : [self.ids2labels[_id] for _id in batch["labels"]]} 

        def predict(batch):
            outputs = self.classifier(batch["text"], self.labels,
                                      multi_label=False, batch_size=batch_size)
            if isinstance(outputs, dict):
                return {"pred_labels": outputs["labels"][0]}
            else:
                return {"pred_labels": [out["labels"][0] for out in outputs]}
        
        dataset = dataset.map(predict, batched=True, batch_size=batch_size)
        dataset = dataset.map(ids2labels, batched=True, batch_size=batch_size)
        return dataset

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
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        def tokenize(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset = dataset.with_format("torch")
        return dataset
            
    def finetune_model(
        self,
        train_dataset : ds.Dataset,
        validation_dataset: ds.Dataset,
        id2label : dict,
        label2id : dict,
        batch_size : int = 8,
        num_train_epochs : int = 10
    ):
        logging.info(f"XPU: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        
        # load model
        logging.info("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            problem_type="single_label_classification", # problem_type (str, optional) — Problem type for XxxForSequenceClassification models. Can be one of "regression", "single_label_classification" or "multi_label_classification".
            num_labels=len(self.labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        # tokenize and prepare the training dataset for training
        logging.info("Tokenizing and prepare the training dataset for training...")
        self.encoded_train_dataset = self.preprocess_data(train_dataset)
        self.encoded_validation_dataset = self.preprocess_data(validation_dataset)
        logging.info(self.encoded_train_dataset)

        # initialize training args
        args = TrainingArguments(
            f"marquesafonso/ticket_triage_{self.model_name.replace('/','_')}_finetuned",
            save_strategy = "epoch",
            eval_strategy="epoch",
            num_train_epochs=num_train_epochs,
            learning_rate=1e-5,
            warmup_ratio=0.1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
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
            eval_dataset=self.encoded_validation_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # fine tune the model
        self.trainer.train()

