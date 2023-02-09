from abc import ABC, abstractmethod

from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.pipeline import Pipeline

from src.components.common import get_logger


class PipelineExecutor(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(self, runner: LocalDagRunner, pipeline: Pipeline) -> None:
        self._runner = runner
        self._pipeline = pipeline
        self._logger = get_logger(f"{__name__}:{self.__class__.__name__}")

    def execute(self) -> None:
        self._logger.info("Running the execution")
        self._runner.run(pipeline=self._pipeline)
