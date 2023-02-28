import os
from dataclasses import dataclass
from typing import List, Optional

from tfx.components.base.base_node import BaseNode
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.metadata import ConnectionConfigType
from tfx.orchestration.pipeline import Pipeline
from tfx.v1.orchestration.metadata import sqlite_metadata_connection_config
from tfx.v1.proto import EvalArgs, PushDestination, TrainArgs

from src.components.common import EPOCHS_CONFIG_FIELD_NAME
from src.components.component_parameters import EvaluatorParameters, ExampleGenParameters, TrainerParameters
from src.components.custom_evaluator import CustomEvaluator
from src.components.custom_example_gen import CustomExampleGen
from src.components.trainer import MODULE_FILE as TRAINER_MODULE_FILE
from src.components.transform import MODULE_FILE as TRANSFORM_MODULE_FILE
from src.parser import RunnerEnv


@dataclass(frozen=True)
class RunnerDependentPipelineParameters:
    pipeline_root: str
    metadata_connection: Optional[ConnectionConfigType]
    pusher_push_destination: PushDestination

    @staticmethod
    def create(runner_env: RunnerEnv, pipeline_name: str) -> "RunnerDependentPipelineParameters":
        gcs_bucket_name = os.environ["GCS_BUCKET_NAME"]
        gcs_dir_prefix = f"gs://{gcs_bucket_name}/"

        pipeline_root = f"tfx_pipeline_root/{pipeline_name}"
        if runner_env is RunnerEnv.VERTEX_AI:
            pipeline_root = gcs_dir_prefix + pipeline_root

        if runner_env is RunnerEnv.LOCAL:
            metadata_connection = sqlite_metadata_connection_config("metadata/pipeline_name/metadata.db")
        else:
            metadata_connection = None

        push_base_directory = "serving_model_dir"
        if runner_env is RunnerEnv.VERTEX_AI:
            push_base_directory = gcs_dir_prefix + push_base_directory

        return RunnerDependentPipelineParameters(
            pipeline_root=pipeline_root,
            metadata_connection=metadata_connection,
            pusher_push_destination=PushDestination(
                filesystem=PushDestination.Filesystem(base_directory=push_base_directory),
            ),
        )


class PipelineFactory:
    @staticmethod
    def _create_example_gen(example_gen_parameters: ExampleGenParameters) -> CustomExampleGen:
        return CustomExampleGen(
            should_use_local_sample_data=example_gen_parameters.should_use_local_sample_data,
            limit_dataset_size=example_gen_parameters.limit_dataset_size,
        )

    @staticmethod
    def _create_statistics_gen(example_gen: CustomExampleGen) -> StatisticsGen:
        return StatisticsGen(examples=example_gen.outputs["examples"])

    @staticmethod
    def _create_schema_gen(statistics_gen: StatisticsGen) -> SchemaGen:
        return SchemaGen(statistics=statistics_gen.outputs["statistics"])

    @staticmethod
    def _create_transform(
        example_gen: CustomExampleGen,
        schema_gen: SchemaGen,
    ) -> Transform:
        return Transform(
            examples=example_gen.outputs["examples"],
            schema=schema_gen.outputs["schema"],
            module_file=TRANSFORM_MODULE_FILE,
            materialize=True,
        )

    @staticmethod
    def _create_trainer(transform: Transform, trainer_parameters: TrainerParameters) -> Trainer:
        return Trainer(
            transform_graph=transform.outputs["transform_graph"],
            examples=transform.outputs["transformed_examples"],
            module_file=TRAINER_MODULE_FILE,
            train_args=TrainArgs(num_steps=trainer_parameters.train_num_steps),
            eval_args=EvalArgs(num_steps=trainer_parameters.eval_num_steps),
            custom_config={EPOCHS_CONFIG_FIELD_NAME: trainer_parameters.epochs},
        )

    @staticmethod
    def _create_evaluator(trainer: Trainer, evaluator_parameters: EvaluatorParameters) -> CustomEvaluator:
        return CustomEvaluator(
            model=trainer.outputs["model"],
            model_run=trainer.outputs["model_run"],
            metric_name=evaluator_parameters.metric_name,
            metric_threshold=evaluator_parameters.metric_threshold,
        )

    @staticmethod
    def _create_pusher(trainer: Trainer, evaluator: CustomEvaluator, push_destination: PushDestination) -> Pusher:
        return Pusher(
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            push_destination=push_destination,
        )

    @staticmethod
    def create(
        pipeline_name: str,
        runner_env: RunnerEnv,
        example_gen_parameters: ExampleGenParameters,
        trainer_parameters: TrainerParameters,
        evaluator_parameters: EvaluatorParameters,
    ) -> Pipeline:
        runner_dependent_pipeline_params = RunnerDependentPipelineParameters.create(
            runner_env=runner_env,
            pipeline_name=pipeline_name,
        )

        example_gen = PipelineFactory._create_example_gen(example_gen_parameters=example_gen_parameters)
        statistics_gen = PipelineFactory._create_statistics_gen(example_gen=example_gen)
        schema_gen = PipelineFactory._create_schema_gen(statistics_gen=statistics_gen)
        transform = PipelineFactory._create_transform(example_gen=example_gen, schema_gen=schema_gen)
        trainer = PipelineFactory._create_trainer(transform=transform, trainer_parameters=trainer_parameters)
        evaluator = PipelineFactory._create_evaluator(trainer=trainer, evaluator_parameters=evaluator_parameters)
        pusher = PipelineFactory._create_pusher(
            trainer=trainer,
            evaluator=evaluator,
            push_destination=runner_dependent_pipeline_params.pusher_push_destination,
        )

        components: List[BaseNode] = [example_gen, statistics_gen, schema_gen, transform, trainer, evaluator, pusher]

        return Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=runner_dependent_pipeline_params.pipeline_root,
            metadata_connection_config=runner_dependent_pipeline_params.metadata_connection,
            components=components,
        )
