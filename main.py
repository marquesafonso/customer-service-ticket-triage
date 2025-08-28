import polars  as pl
import warnings
from src.dataset import load_dataset, process_dataset
from src.models import TicketTriageModel

def main():
    warnings.filterwarnings("ignore") 
    dataset = load_dataset()[:1000]
    print(dataset.height)

    queue_labels = dataset.select("queue").unique().to_series().to_list()

    dataset = dataset.with_columns((pl.concat_str([pl.col("subject") + "\n", pl.col("body")]).alias("ticket")))
    print(dataset)
    model = TicketTriageModel(labels=queue_labels)

    # test_str = dataset.select("ticket").to_series().to_list()[0]
    # print(model.process_ticket(customer_ticket=test_str))
    
    dataset = process_dataset(
        model=model,
        dataset=dataset,
        batch_size=200,
        chunk_size=500
    )

    dataset.write_parquet("dataset_with_preds.parquet")
    print(dataset)
    accuracy = dataset.filter((pl.col("queue") == pl.col("pred_queue_category"))).height * 100 / dataset.height
    print(accuracy)


if __name__ == "__main__":
    main()
