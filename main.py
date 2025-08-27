import polars  as pl
import warnings
from src.dataset import load_dataset
from src.models import TicketTriageModel

def main():
    warnings.filterwarnings("ignore") 
    # Download latest version
    dataset = load_dataset()[:3]
    print(dataset)
    print(dataset.schema)
    queue_labels = dataset.select("queue").unique().to_series().to_list()
    print(queue_labels)
    customer_email = """
    Subject: URGENT - Cannot access my account after payment

    I paid for the premium plan 3 hours ago and still can't access any features.
    I've tried logging out and back in multiple times. This is unacceptable as I
    have a client presentation in an hour and need the analytics dashboard.
    Please fix this immediately or refund my payment.
    """
    dataset = dataset.with_columns((pl.concat_str([pl.col("subject") + "\n", pl.col("body")]).alias("ticket")))
    print(dataset)
    test_str = dataset.select("ticket").to_series().to_list()[0]
    print(test_str)
    model = TicketTriageModel()
    dataset = dataset.with_columns(((pl.col("ticket").map_elements(model.process_ticket)).cast(pl.String)).alias("pred_queue_category"))
    print(dataset)
    accuracy = dataset.filter((pl.col("queue") == pl.col("pred_queue_category"))).count() * 100 // dataset.count()
    print(accuracy)


if __name__ == "__main__":
    main()
