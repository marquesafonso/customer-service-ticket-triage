import warnings
import datasets as ds
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import trackio

class TicketTriageModel:
    def __init__(self,
                labels: list,
                model_name = "microsoft/deberta-v3-base"):
        warnings.filterwarnings("ignore")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    def __init__(self,
                labels: list,
                model_name = "microsoft/deberta-v3-base"):
        warnings.filterwarnings("ignore")
        self.model_name = model_name
        self.labels = labels    
    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        self.metrics = {
            'f1': f1_micro_average,
            'roc_auc': roc_auc,
            'accuracy': accuracy
            }
        return self.metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        self.result = self.multi_label_metrics(predictions=preds, labels=p.label_ids)
        return self.result
    
    def preprocess_data(self, dataset):
        # take a batch of texts
        text = list(dataset["ticket"])
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = list(dataset["queue_encoded"])
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, _ in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[idx]

        encoding["labels"] = labels_matrix.tolist()

        return encoding
            
    def finetune_model(self,
                       train_dataset,
                       id2label,
                       label2id,
                       batch_size = 8,
                       metric_name = "f1"):
        # load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(self.labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # tokenize and prepare the training dataset for training
        self.encoded_train_dataset = self.preprocess_data(train_dataset)
        
        # initialize training args
        args = TrainingArguments(
            f"marquesafonso/ticket_triage_{self.model_name.replace('/','_')}_finetuned",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            metric_for_best_model=metric_name,
            label_names = self.labels,
            report_to="trackio",
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

