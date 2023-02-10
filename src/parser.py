import argparse
from dataclasses import dataclass

from src.components.common import get_logger


@dataclass(frozen=True)
class CommandLineArgs:
    should_use_local_sample_data: bool
    pipeline_name: str
    limit_dataset_size: int
    train_num_steps: int
    eval_num_steps: int
    epochs: int
    metric_name: str
    metric_threshold: float


class Parser:
    @staticmethod
    def _create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--use-local-sample-data",
            action="store_true",
            help="Specifies source of the data used. When the flag is provided, a small sample stored locally in the "
            "repo will be used. Otherwise the data will get fetched from Kaggle. Note: In order to fetch from "
            "Kaggle API, credentials have to be stored in the ~/.kaggle/kaggle.json file",
        )
        parser.add_argument("--pipeline-name", type=str, default="pipeline_name", help="Name of the pipeline")
        parser.add_argument(
            "--limit-dataset-size",
            type=int,
            default=None,
            help="Keeps only first X entries from the dataset. Full dataset will be used when `None`",
        )
        parser.add_argument(
            "--train-num-steps", type=int, default=100, help="Defines for how many steps training data will be split"
        )
        parser.add_argument(
            "--eval-num-steps", type=int, default=50, help="Defines for how many steps eval data will be split"
        )
        parser.add_argument("--epochs", type=int, default=10, help="For how many epochs the model will be trained")
        parser.add_argument(
            "--metric-name",
            type=str,
            default="epoch_factorized_top_k/top_5_categorical_accuracy",
            help="Which metric will be used to bless/not bless the model",
        )
        parser.add_argument(
            "--metric-threshold",
            type=float,
            default=0.02,
            help="Minimal threshold for the metric specified above in order to bless the model",
        )
        return parser

    @classmethod
    def parse(cls) -> CommandLineArgs:
        logger = get_logger(f"{__name__}:{cls.__name__}")
        logger.info("Parsing arguments")

        parser = Parser._create_parser()
        parsed_args = parser.parse_args()
        args = CommandLineArgs(
            should_use_local_sample_data=parsed_args.use_local_sample_data,
            pipeline_name=parsed_args.pipeline_name,
            limit_dataset_size=parsed_args.limit_dataset_size,
            train_num_steps=parsed_args.train_num_steps,
            eval_num_steps=parsed_args.eval_num_steps,
            epochs=parsed_args.epochs,
            metric_name=parsed_args.metric_name,
            metric_threshold=parsed_args.metric_threshold,
        )
        logger.info(f"Parsed arguments: {args}")
        return args
