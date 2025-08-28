import warnings
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

    def process_ticket(self, customer_ticket: str):
        output = self.classifier(customer_ticket, self.labels, multi_label=False)["labels"][0]
        return output
    
    def process_batch(self, tickets: list[str]) -> list[str]:
        """
        Run batch inference on a list of tickets.
        """
        outputs = self.classifier(
            tickets,
            self.labels,
            multi_label=False
        )
        return [out["labels"][0] for out in outputs]

