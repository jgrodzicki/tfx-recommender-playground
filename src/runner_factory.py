from tfx.orchestration.kubeflow.v2.kubeflow_v2_dag_runner import KubeflowV2DagRunner, KubeflowV2DagRunnerConfig
from tfx.orchestration.local.local_dag_runner import LocalDagRunner


class RunnerFactory:
    @staticmethod
    def create_local_dag_runner() -> LocalDagRunner:
        return LocalDagRunner()  # type: ignore[no-untyped-call]

    @staticmethod
    def create_kubeflow_dag_runner(output_filename: str, google_cloud_project: str) -> KubeflowV2DagRunner:
        artifact_registry_region = "europe-central2"
        repo_name = "tfx-recommender-playground"
        image_name = "image"
        tag = "master"
        image = f"{artifact_registry_region}-docker.pkg.dev/{google_cloud_project}/{repo_name}/{image_name}:{tag}"

        return KubeflowV2DagRunner(
            config=KubeflowV2DagRunnerConfig(default_image=image),
            output_filename=output_filename,
        )
