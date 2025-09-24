import warnings, logging, os
from typing import Any
import datasets as ds
from transformers import (
    pipeline, 
    get_scheduler,
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
from dotenv import load_dotenv
from src.dataset import oversample_with_interleave

class TicketTriageModel:
    def __init__(
        self,
        labels: list,
        model_name = "microsoft/deberta-v3-base",
        id2label: dict[int, Any] = None
    ):
        warnings.filterwarnings("ignore")
        logging.basicConfig(
           level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
        torch.cuda.empty_cache() if torch.cuda.is_available() else torch.xpu.empty_cache() if torch.xpu.is_available() else "pass"
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
            hypothesis_template = "This example is about {}"
            outputs = self.classifier(batch["text"], self.labels,
                                      multi_label=False, hypothesis_template=hypothesis_template, batch_size=batch_size)
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
        model_name :str = "microsoft/deberta-v3-base"
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
            # https://huggingface.co/docs/transformers/pad_truncation
            return self.tokenizer(examples["text"], padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
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
        num_train_epochs : int = 5,
        oversample : bool = False
    ):
        load_dotenv()
        HF_USER = os.getenv("HF_USER")
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
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.max_length = self.config.max_position_embeddings
        logging.info(f"Max Length: {self.max_length}")

        if oversample:
            ## Oversampling (underrepresented) classes
            logging.info("Oversampling (underrepresented) classes...")
            train_dataset = oversample_with_interleave(
                train_dataset,
                num_labels=len(self.labels),
                seed=10
            )

        ## Tokenize and prepare the training dataset for training
        logging.info("Tokenizing and prepare the training dataset for training...")
        self.encoded_train_dataset = self.preprocess_data(train_dataset)
        self.encoded_validation_dataset = self.preprocess_data(validation_dataset)
        logging.info(self.encoded_train_dataset)

        ## Initialize training args
        args = TrainingArguments(
            f"{HF_USER}/ticket_triage_{self.model_name.replace('/','_')}_finetuned",
            save_strategy = "epoch",
            eval_strategy="epoch",
            num_train_epochs=num_train_epochs,
            warmup_ratio=0.06,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2, ## Efective batch size will be batch_size * gradient_accumulation_steps
            max_grad_norm=0.5,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            load_best_model_at_end=True,
            lr_scheduler_type="linear",
            report_to="trackio",
            remove_unused_columns=False,
            push_to_hub=True
        )

        ## Custom optimizer an lr_scheduler
        ## Following hyperparams from: https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=9e-06,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01
        )
        num_training_steps = len(self.encoded_train_dataset) // batch_size * num_train_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=0.06*num_training_steps
        )

        ## Initialize the trainer
        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=self.encoded_train_dataset,
            eval_dataset=self.encoded_validation_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # fine tune the model
        self.trainer.train()

