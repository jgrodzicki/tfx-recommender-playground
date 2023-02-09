from abc import ABC, abstractmethod

from tfx.orchestration.pipeline import Pipeline

from src.components.common import get_logger
from src.runner_factory import RunnerFactory


class PipelineExecutor(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self._runner = RunnerFactory.create_local_dag_runner()
        self._logger = get_logger(f"{__name__}:{self.__class__.__name__}")

    def execute(self) -> None:
        self._logger.info("Running the execution")
        self._runner.run(pipeline=self._pipeline)
