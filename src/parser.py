import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandLineArgs:
    pipeline_name: str
    pipeline_root: str
    metadata_path: str
    limit_dataset_size: int
    train_num_steps: int
    eval_num_steps: int
    epochs: int
    metric_name: str
    metric_threshold: float
    serving_model_dir: str


class Parser:
    @staticmethod
    def _create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--pipeline-name", type=str, default="pipeline_name", help="Name of the pipeline")
        parser.add_argument(
            "--pipeline-root", type=str, default="pipeline_root", help="Folder under which pipeline data will be stored"
        )
        parser.add_argument(
            "--metadata-path",
            type=str,
            default="metadata/pipeline_name/metadata.db",
            help="Path to the file where metadata will be stored",
        )
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
        parser.add_argument(
            "--serving-model-dir",
            type=str,
            default="serving_model_dir",
            help="Directory where model will be pushed if good enough for serving",
        )
        return parser

    @staticmethod
    def parse() -> CommandLineArgs:
        parser = Parser._create_parser()
        parsed_args = parser.parse_args()
        return CommandLineArgs(
            pipeline_name=parsed_args.pipeline_name,
            pipeline_root=parsed_args.pipeline_root,
            metadata_path=parsed_args.metadata_path,
            limit_dataset_size=parsed_args.limit_dataset_size,
            train_num_steps=parsed_args.train_num_steps,
            eval_num_steps=parsed_args.eval_num_steps,
            epochs=parsed_args.epochs,
            metric_name=parsed_args.metric_name,
            metric_threshold=parsed_args.metric_threshold,
            serving_model_dir=parsed_args.serving_model_dir,
        )
