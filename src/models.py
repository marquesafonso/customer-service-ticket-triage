import warnings
import datasets as ds
from transformers import pipeline
import torch

class TicketTriageModel:
    def __init__(self,
                labels: list,
                model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        warnings.filterwarnings("ignore")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline("zero-shot-classification", model=self.model_name, device=self.device)
        self.labels = labels

    def process_dataset(self, dataset: ds.Dataset, batch_size: int = 32) -> ds.Dataset:
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

