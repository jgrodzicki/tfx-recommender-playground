import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from google.cloud import storage  # type: ignore[attr-defined]
from tensorflow.core.util import event_pb2
from tfx.components.evaluator import constants as evaluator_constants
from tfx.dsl.components.base import base_component, base_executor, executor_spec
from tfx.types import artifact_utils, standard_artifacts
from tfx.types.artifact import Artifact
from tfx.types.channel import BaseChannel, Channel
from tfx.types.component_spec import ChannelParameter, ComponentSpec, ExecutionParameter
from tfx.types.standard_component_specs import BLESSING_KEY, MODEL_KEY, MODEL_RUN_KEY
from tfx.utils import io_utils

from src.components.common import get_logger


class NoMetricFoundError(Exception):
    pass


METRIC_THRESHOLD_FIELD = "metric_threshold"
METRIC_NAME_FIELD = "metric_name"


class Executor(base_executor.BaseExecutor):
    def __init__(self, context: Any) -> None:
        super().__init__(context=context)
        self._logger = get_logger(f"{__name__}:{self.__class__.__name__}")

    @staticmethod
    def _get_single_artifact(artifacts: List[Artifact]) -> Artifact:
        return artifact_utils.get_single_instance(artifacts)

    def _retrieve_latest_eval_metric_from_local(self, validation_folder_path: str, metric_name: str) -> float:
        metrics_files = list(pathlib.Path(validation_folder_path).iterdir())
        self._logger.info(f"Found following local files with metrics: {metrics_files}")
        metric = self._get_latest_metric_from_files(metric_files=metrics_files, metric_name=metric_name)
        return metric

    def _retrieve_latest_eval_metric_from_gcs(self, validation_folder_path: str, metric_name: str) -> float:
        uri_split = validation_folder_path.split("/")
        bucket_name = uri_split[2]
        prefix = "/".join(uri_split[3:])

        client = storage.Client()
        bucket = client.get_bucket(bucket_or_name=bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        blob_list = list(blobs)  # Convert to avoid `Iterator has already started` error
        blob_list = blob_list[1:]  # Ugly way to get rid of the folder listed as a blob
        self._logger.info(f"Found following files in bucket: {bucket.name} - {[b.name for b in blob_list]}")

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            metric_files = []
            for blob in blob_list:
                self._logger.info(f"Downloading: {blob.name}")
                filename = blob.name.replace("/", "_")
                path = f"{tmp_dir_name}/{filename}"

                blob.download_to_filename(path)
                metric_files.append(path)

            metric = self._get_latest_metric_from_files(metric_files=metric_files, metric_name=metric_name)
        return metric

    def _retrieve_latest_eval_metric(self, model_run_uri: str, metric_name: str) -> float:
        self._logger.info(f"Retrieving model metrics for {model_run_uri}")
        validation_folder_path = f"{model_run_uri}/validation"
        if model_run_uri.startswith("gs://"):
            return self._retrieve_latest_eval_metric_from_gcs(
                validation_folder_path=validation_folder_path,
                metric_name=metric_name,
            )
        else:
            return self._retrieve_latest_eval_metric_from_local(
                validation_folder_path=validation_folder_path,
                metric_name=metric_name,
            )

    def _get_latest_metric_from_files(
        self,
        metric_files: Union[List[str], List[pathlib.Path]],
        metric_name: str,
    ) -> float:
        self._logger.info(f"Getting evaluation metric: {metric_name}")
        last_eval_metric: Optional[float] = None
        metrics_dataset = tf.data.TFRecordDataset(metric_files)

        for serialized_metric in metrics_dataset:
            event = event_pb2.Event.FromString(serialized_metric.numpy())
            for metric in event.summary.value:
                if metric.tag == metric_name:
                    last_eval_metric = tf.make_ndarray(metric.tensor).item()

        if last_eval_metric is None:
            raise NoMetricFoundError(f"Following metric wasn't found for the trained model: {metric_name}")

        self._logger.info(f"Found metric: {metric_name} - {last_eval_metric}")
        return last_eval_metric

    def _should_bless_model(self, metrics_threshold: float, latest_eval_metric: float) -> bool:
        should_bless = latest_eval_metric >= metrics_threshold
        self._logger.info(f"Minimal threshold: {metrics_threshold}. Model {'IS' if should_bless else 'ISNT'} blessed")
        return should_bless

    def _write_blessing_result(
        self,
        blessing_artifact: Artifact,
        model_artifact: Artifact,
        should_bless_model: bool,
    ) -> None:
        self._logger.info("Writing blessing results to the artifacts")
        if should_bless_model:
            filename = evaluator_constants.BLESSED_FILE_NAME
            artifact_property_value = evaluator_constants.BLESSED_VALUE
        else:
            filename = evaluator_constants.NOT_BLESSED_FILE_NAME
            artifact_property_value = evaluator_constants.NOT_BLESSED_VALUE

        io_utils.write_string_file(
            file_name=os.path.join(blessing_artifact.uri, filename),
            string_value="",
        )
        blessing_artifact.set_int_custom_property(
            key=evaluator_constants.ARTIFACT_PROPERTY_BLESSED_KEY,
            value=artifact_property_value,
        )

        blessing_artifact.set_string_custom_property(
            key=evaluator_constants.ARTIFACT_PROPERTY_CURRENT_MODEL_URI_KEY,
            value=model_artifact.uri,
        )
        blessing_artifact.set_int_custom_property(
            key=evaluator_constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY,
            value=model_artifact.id,
        )

    def Do(
        self,
        input_dict: Dict[str, List[Artifact]],
        output_dict: Dict[str, List[Artifact]],
        exec_properties: Dict[str, Any],
    ) -> None:
        model_run_artifact = self._get_single_artifact(artifacts=input_dict[MODEL_RUN_KEY])
        latest_eval_metric = self._retrieve_latest_eval_metric(
            model_run_uri=model_run_artifact.uri,
            metric_name=exec_properties[METRIC_NAME_FIELD],
        )

        should_bless_model = self._should_bless_model(
            metrics_threshold=exec_properties[METRIC_THRESHOLD_FIELD],
            latest_eval_metric=latest_eval_metric,
        )
        self._write_blessing_result(
            blessing_artifact=self._get_single_artifact(artifacts=output_dict[BLESSING_KEY]),
            model_artifact=self._get_single_artifact(artifacts=input_dict[MODEL_KEY]),
            should_bless_model=should_bless_model,
        )


class CustomEvaluatorSpec(ComponentSpec):  # type: ignore[no-untyped-call]
    PARAMETERS = {
        METRIC_THRESHOLD_FIELD: ExecutionParameter(type=float),  # type: ignore[no-untyped-call]
        METRIC_NAME_FIELD: ExecutionParameter(type=str),  # type: ignore[no-untyped-call]
    }
    INPUTS = {
        MODEL_KEY: ChannelParameter(type=standard_artifacts.Model),
        MODEL_RUN_KEY: ChannelParameter(type=standard_artifacts.ModelRun),
    }
    OUTPUTS = {
        BLESSING_KEY: ChannelParameter(type=standard_artifacts.ModelBlessing),
    }


class CustomEvaluator(base_component.BaseComponent):
    SPEC_CLASS = CustomEvaluatorSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(
        self,
        metric_threshold: float,
        metric_name: str,
        model: BaseChannel,
        model_run: BaseChannel,
    ) -> None:
        spec = CustomEvaluatorSpec(
            **{
                METRIC_THRESHOLD_FIELD: metric_threshold,
                METRIC_NAME_FIELD: metric_name,
                MODEL_KEY: model,
                MODEL_RUN_KEY: model_run,
                BLESSING_KEY: Channel(type=standard_artifacts.ModelBlessing),
            }
        )  # type: ignore[no-untyped-call]
        super().__init__(spec=spec)
