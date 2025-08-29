from src.dataset import load_dataset
from src.models import TicketTriageModel

def main():
    dataset, queue_labels = load_dataset()
    dataset = dataset.take(300)

    model_name = "microsoft/deberta-xlarge"
    model = TicketTriageModel(labels=queue_labels)

    dataset = model.process_dataset(dataset, batch_size=256)

    dataset.to_parquet(f"dataset_preds_{model_name.replace('/', '_')}.parquet")

    accuracy = dataset.filter(lambda x: x["queue"] == x["pred_queue"]).num_rows * 100 / dataset.num_rows
    print(accuracy)


if __name__ == "__main__":
    main()
