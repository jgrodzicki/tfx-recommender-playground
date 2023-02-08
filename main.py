from tfx.orchestration.metadata import sqlite_metadata_connection_config

from src.components.component_parameters import (
    EvaluatorParameters,
    ExampleGenParameters,
    PusherParameters,
    TrainerParameters,
)
from src.executor import LocalExecutor
from src.parser import Parser
from src.pipeline import PipelineFactory
from src.runner_factory import RunnerFactory


def main() -> None:
    args = Parser.parse()
    runner = RunnerFactory.create_local_dag_runner()
    pipeline = PipelineFactory.create(
        pipeline_name=args.pipeline_name,
        pipeline_root=args.pipeline_root,
        metadata_connection_config=sqlite_metadata_connection_config(args.metadata_path),
        example_gen_parameters=ExampleGenParameters(
            should_use_local_sample_data=args.should_use_local_sample_data,
            limit_dataset_size=args.limit_dataset_size,
        ),
        trainer_parameters=TrainerParameters(
            epochs=args.epochs,
            train_num_steps=args.train_num_steps,
            eval_num_steps=args.eval_num_steps,
        ),
        evaluator_parameters=EvaluatorParameters(
            metric_name=args.metric_name,
            metric_threshold=args.metric_threshold,
        ),
        pusher_parameters=PusherParameters(serving_model_dir=args.serving_model_dir),
    )
    executor = LocalExecutor(runner=runner, pipeline=pipeline)
    executor.execute()


if __name__ == "__main__":
    main()
