from abc import ABC, abstractmethod

from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.pipeline import Pipeline


class Executor(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass


class LocalExecutor(Executor):
    def __init__(self, runner: LocalDagRunner, pipeline: Pipeline) -> None:
        self._runner = runner
        self._pipeline = pipeline

    def execute(self) -> None:
        self._runner.run(pipeline=self._pipeline)
