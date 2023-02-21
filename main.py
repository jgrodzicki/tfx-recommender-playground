from src.components.component_parameters import EvaluatorParameters, ExampleGenParameters, TrainerParameters
from src.parser import Parser
from src.pipeline_executor import PipelineExecutorFactory
from src.pipeline_factory import PipelineFactory


def main() -> None:
    args = Parser.parse()
    pipeline = PipelineFactory.create(
        pipeline_name=args.pipeline_name,
        runner_env=args.runner_env,
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
    )
    executor = PipelineExecutorFactory.create(runner_env=args.runner_env, pipeline=pipeline)
    executor.execute()


if __name__ == "__main__":
    main()
