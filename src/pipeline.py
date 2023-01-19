from src.components.custom_example_gen import CustomExampleGen


class PipelineFactory:
    @staticmethod
    def _create_example_gen() -> CustomExampleGen:
        return CustomExampleGen()
