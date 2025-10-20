import os, logging
from src.dataset import load_dataset
from src.models import TicketTriageModel, BaseModel
from huggingface_hub import login
from dotenv import load_dotenv

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
        num_train_epochs=5,
        oversample=False
    )

    ## Evaluate on test set
    logging.info("Evaluating on test set...")
    finetuned_model_name = f"{HF_USER}/ticket_triage_{basemodel_name.replace('/',"_")}_finetuned"
    finetuned_model = TicketTriageModel(
        model_name=finetuned_model_name,
        labels=queue_labels,
        id2label=id2label,
        token=HF_TOKEN
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
