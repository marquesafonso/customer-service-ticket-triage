import os
from src.dataset import load_dataset
from src.models import TicketTriageModel, BaseModel
from huggingface_hub import login
from dotenv import load_dotenv

## TODO: improve result evaluation with compute_metrics

def main():
    # log in to HF
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    ## Prepare datasets
    dataset, queue_labels, id2labels, labels2ids = load_dataset()
    test_dataset, train_dataset = dataset["test"], dataset["train"]
    
    ## Prepare base model
    basemodel_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    basemodel = BaseModel(model_name=basemodel_name, labels=queue_labels)

    # Tokenize train_dataset for training
    encoded_train_dataset = train_dataset.map(basemodel.preprocess_data, batched=True, remove_columns=train_dataset.column_names)

    ## finetune the model
    basemodel.finetune_model(encoded_train_dataset, id2labels, labels2ids)

    ## Evaluate
    finetuned_model_name = f"marquesafonso/ticket_triage_{basemodel_name.split('/')[1]}_finetuned"
    finetuned_model = TicketTriageModel(model_name=finetuned_model_name, labels=queue_labels)

    finetuned_dataset = finetuned_model.get_predictions_from_dataset(test_dataset, batch_size=256)
    finetuned_dataset.to_parquet(f"finetuned_preds.parquet")

    finetuned_accuracy = finetuned_dataset.filter(lambda x: x["queue"] == x["pred_queue"]).num_rows * 100 / finetuned_dataset.num_rows
    print(finetuned_accuracy)


if __name__ == "__main__":
    main()
