import outlines
import ast
from enum import Enum
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


class QueueCategory(str, Enum):
    SalesAndPreSales = "Sales and Pre-Sales"
    TechnicalSupport = "Technical Support"
    ITSupport = "IT Support"
    ServiceOutagesAndMaintenance = "Service Outages and Maintenance"
    BilligandPayments = "Billing and Payments"
    GeneralInquiry = "General Inquiry"
    CustomerService = "Customer Service"
    ProductSupport = "Product Support"
    ReturnsAndExchanges = "Returns and Exchanges"
    HumanResources = "Human Resources"

class ServiceTicket(BaseModel):
    category: QueueCategory



class TicketTriageModel:
    def __init__(self, model_name = "google/gemma-3-270m"):
        load_dotenv()
        self.model_name = model_name
        self.model = outlines.from_transformers(
            AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto"),
            AutoTokenizer.from_pretrained(self.model_name)
        )
    def _preprocess_prompt(self, customer_ticket: str):
        prompt = f"""
        <|im_start|>user
        Analyze this customer email:

        {customer_ticket}
        <|im_end|>
        <|im_start|>assistant
        """
        return prompt

    def process_ticket(self, customer_ticket: str):
        queue_category = self.model(
            self._preprocess_prompt(customer_ticket),
            ServiceTicket,
            max_new_tokens=500
        )
        return ast.literal_eval(queue_category)["category"]

