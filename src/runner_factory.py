from tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner import KubeflowV2DagRunner, KubeflowV2DagRunnerConfig
from tfx.orchestration.local.local_dag_runner import LocalDagRunner


class RunnerFactory:
    @staticmethod
    def create_local_dag_runner() -> LocalDagRunner:
        return LocalDagRunner()  # type: ignore[no-untyped-call]

    @staticmethod
    def create_kubeflow_dag_runner(
        output_filename: str,
        google_cloud_project: str,
        gcp_artifact_registry_docker_tag: str,
    ) -> KubeflowV2DagRunner:
        artifact_registry_region = "europe-central2"
        repo_name = "tfx-recommender-playground"
        image_name = "image"

        docker_image = (
            f"{artifact_registry_region}-docker.pkg.dev/{google_cloud_project}/{repo_name}/"
            f"{image_name}:{gcp_artifact_registry_docker_tag}"
        )

        return KubeflowV2DagRunner(
            config=KubeflowV2DagRunnerConfig(default_image=docker_image),
            output_filename=output_filename,
        )
