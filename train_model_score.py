'''
To train model and check score
'''
from training import check_scoring,train_test_model
from training.data import get_train_test_data
import argparse
import logging


def go(args):
    """
    to execute pipeline with arguments
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Getting train test data")
    train,test = get_train_test_data()

    if args.action == "all" or args.action == "train_test_model":
        logging.info("Train Test Model")
        train_test_model(train,test)

    if args.action == "all" or args.action == "scoring":
        logging.info("Slicing Scroing")
        check_scoring(test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["train_test_model",
                 "scoring",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    args = parser.parse_args()

    go(args)
