from src.dataset import load_dataset
from src.models import TicketTriageModel

def main():
    dataset, queue_labels = load_dataset()
    test_dataset, train_dataset = dataset["test"], dataset["train"]

    model_name = "microsoft/deberta-xlarge"
    model = TicketTriageModel(labels=queue_labels)

    test_dataset = model.process_dataset(test_dataset, batch_size=256)

    test_dataset.to_parquet(f"dataset_preds_{model_name.replace('/', '_')}_test.parquet")

    accuracy = test_dataset.filter(lambda x: x["queue"] == x["pred_queue"]).num_rows * 100 / dataset.num_rows
    print(accuracy)


if __name__ == "__main__":
    main()
