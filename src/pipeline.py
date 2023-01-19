from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen

from src.components.custom_example_gen import CustomExampleGen


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
