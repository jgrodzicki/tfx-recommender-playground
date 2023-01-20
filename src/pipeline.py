from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.v1.proto import EvalArgs, TrainArgs

from src.components.consts import EPOCHS_CONFIG_FIELD_NAME
from src.components.custom_example_gen import CustomExampleGen
from src.components.trainer import MODULE_FILE as trainer_module_file
from src.components.transform import MODULE_FILE as transform_module_file


class PipelineFactory:
    @staticmethod
    def _create_example_gen() -> CustomExampleGen:
        return CustomExampleGen()

    @staticmethod
    def _create_statistics_gen(example_gen: CustomExampleGen) -> StatisticsGen:
        return StatisticsGen(examples=example_gen.outputs["examples"])

    @staticmethod
    def _create_schema_gen(statistics_gen: StatisticsGen) -> SchemaGen:
        return SchemaGen(statistics=statistics_gen.outputs["statistics"])

    @staticmethod
    def _create_transform(example_gen: CustomExampleGen, schema_gen: SchemaGen) -> Transform:
        return Transform(
            examples=example_gen.outputs["examples"],
            schema=schema_gen.outputs["schema"],
            module_file=transform_module_file,
            materialize=True,
        )

    @staticmethod
    def _create_trainer(transform: Transform) -> Trainer:
        return Trainer(
            transform_graph=transform.outputs["transform_graph"],
            examples=transform.outputs["transformed_examples"],
            train_args=TrainArgs(num_steps=10),
            eval_args=EvalArgs(num_steps=10),
            module_file=trainer_module_file,
            custom_config={EPOCHS_CONFIG_FIELD_NAME: 10},
        )
