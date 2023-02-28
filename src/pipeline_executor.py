import os
from abc import ABC, abstractmethod

from google.cloud import aiplatform
from tfx.orchestration.pipeline import Pipeline

from src.components.common import get_logger
from src.parser import RunnerEnv
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


class VertexAIPipelineExecutor(PipelineExecutor):
    def __init__(
        self,
        pipeline: Pipeline,
        google_cloud_project: str,
        google_cloud_region: str,
        gcp_artifact_registry_docker_tag: str,
    ) -> None:
        self._runner = RunnerFactory.create_kubeflow_dag_runner(
            output_filename=pipeline.pipeline_name + "_pipeline.json",
            google_cloud_project=google_cloud_project,
            gcp_artifact_registry_docker_tag=gcp_artifact_registry_docker_tag,
        )
        self._pipeline = pipeline
        self._google_cloud_project = google_cloud_project
        self._google_cloud_region = google_cloud_region
        self._logger = get_logger(f"{__name__}:{self.__class__.__name__}")

    def execute(self) -> None:
        self._runner.run(self._pipeline)  # Will write a pipeline definition to self._runner._output_filename

        aiplatform.init(project=self._google_cloud_project, location=self._google_cloud_region)
        job = aiplatform.pipeline_jobs.PipelineJob(
            template_path=self._runner._output_filename,
            display_name=self._pipeline.pipeline_name,
        )
        job.submit()


class UnhandledRunnerEnv(Exception):
    pass


class PipelineExecutorFactory:
    @staticmethod
    def create(runner_env: RunnerEnv, pipeline: Pipeline) -> PipelineExecutor:
        if runner_env is RunnerEnv.LOCAL:
            return LocalPipelineExecutor(pipeline=pipeline)

        elif runner_env is RunnerEnv.VERTEX_AI:
            google_cloud_project = os.environ["GOOGLE_CLOUD_PROJECT"]
            google_cloud_region = os.environ["CLOUD_ML_REGION"]
            gcp_artifact_registry_docker_tag = os.environ["GCP_ARTIFACT_REGISTRY_DOCKER_TAG"]

            return VertexAIPipelineExecutor(
                pipeline=pipeline,
                google_cloud_project=google_cloud_project,
                google_cloud_region=google_cloud_region,
                gcp_artifact_registry_docker_tag=gcp_artifact_registry_docker_tag,
            )

        else:
            raise UnhandledRunnerEnv(f"Unhandled runner env: {runner_env}")
