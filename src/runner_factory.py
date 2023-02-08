from tfx.orchestration.local.local_dag_runner import LocalDagRunner


class RunnerFactory:
    @staticmethod
    def create_local_dag_runner() -> LocalDagRunner:
        return LocalDagRunner()  # type: ignore[no-untyped-call]
