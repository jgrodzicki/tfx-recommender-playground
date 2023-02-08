import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tensorflow as tf
from tfx.dsl.components.base import base_component, base_executor, executor_spec
from tfx.types import artifact_utils, standard_artifacts
from tfx.types.artifact import Artifact, Property, PropertyType
from tfx.types.channel import Channel
from tfx.types.component_spec import ChannelParameter, ComponentSpec, ExecutionParameter
from tfx.types.standard_artifacts import _TfxArtifact
from tqdm import tqdm

from src.components.common import REQUIRED_COLUMNS

KAGGLE_DATASET = "shuyangli94/food-com-recipes-and-user-interactions"
RAW_RECIPES_FILENAME = Path("RAW_recipes.csv")
RAW_INTERACTIONS_FILENAME = Path("RAW_interactions.csv")

KAGGLE_DATA_FOLDER = Path("data")
SAMPLE_DATA_FOLDER = Path("sample_data")

TRAIN_SET_SIZE_FRAC = 0.7
TRAIN_SPLIT_NAME = "train"
EVAL_SPLIT_NAME = "eval"


SIZE_PROPERTY = Property(type=PropertyType.INT)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class SpecFields:
    SHOULD_USE_LOCAL_SAMPLE_DATA = "should_use_local_sample_data"
    LIMIT_DATASET_SIZE = "limit_dataset_size"
    EXAMPLES = "examples"
    SIZE = "size"


def _dict_to_example_single(instance: Dict[str, Any]) -> tf.train.Example:
    feature = {}
    for key, value in instance.items():
        if value is None:
            feature[key] = tf.train.Feature()
        elif isinstance(value, int):
            feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        else:
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(str(value), "utf-8")]))

    return tf.train.Example(features=tf.train.Features(feature=feature))


class Executor(base_executor.BaseExecutor):
    @staticmethod
    def _download_kaggle_files() -> None:
        import kaggle  # Local import as authentication is performed on import - would fail if no creds provided

        for filename in [RAW_RECIPES_FILENAME, RAW_INTERACTIONS_FILENAME]:
            kaggle.api.dataset_download_file(
                dataset=KAGGLE_DATASET,
                file_name=str(filename),
                path=str(KAGGLE_DATA_FOLDER),
            )

    @staticmethod
    def _read_file(folder: Path, filename: Path) -> pd.DataFrame:
        return pd.read_csv(str(folder / filename) + ".zip")

    @staticmethod
    def _merge_and_clean_dfs(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        merged_df_full = recipes_df.merge(right=interactions_df, left_on="id", right_on="recipe_id")
        merged_df = merged_df_full[REQUIRED_COLUMNS].dropna(how="any")
        return merged_df

    @staticmethod
    def _get_clean_data(should_use_local_sample_data: bool) -> pd.DataFrame:
        if should_use_local_sample_data:
            folder = SAMPLE_DATA_FOLDER
        else:
            Executor._download_kaggle_files()
            folder = KAGGLE_DATA_FOLDER

        recipes_df = Executor._read_file(folder=folder, filename=RAW_RECIPES_FILENAME)
        interactions_df = Executor._read_file(folder=folder, filename=RAW_INTERACTIONS_FILENAME)

        data = Executor._merge_and_clean_dfs(recipes_df=recipes_df, interactions_df=interactions_df)
        return data

    @staticmethod
    def _get_train_eval_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = data.sample(frac=TRAIN_SET_SIZE_FRAC)
        eval_df = data.drop(train_df.index)
        return train_df, eval_df

    @staticmethod
    def _get_single_artifact(artifacts: List[Artifact]) -> Artifact:
        return artifact_utils.get_single_instance(artifacts)

    @staticmethod
    def _set_size_artifact(size_artifact: Artifact, train_df_size: int, eval_df_size: int) -> None:
        size_artifact.train = train_df_size
        size_artifact.eval = eval_df_size

    @staticmethod
    def _set_examples_artifact(examples_artifact: Artifact) -> None:
        examples_artifact.split_names = artifact_utils.encode_split_names([TRAIN_SPLIT_NAME, EVAL_SPLIT_NAME])

    @staticmethod
    def _convert_and_write_tfrecords(df: pd.DataFrame, examples_artifact_uri: str, split_name: str) -> None:
        output_folder = f"{examples_artifact_uri}/Split-{split_name}"
        os.makedirs(output_folder)

        writer = tf.io.TFRecordWriter(
            f"{output_folder}/tfrecord.gz",
            options=tf.compat.v1.io.TFRecordCompressionType.GZIP,
        )

        tqdm_desc = f"Converting to TFRecord for split - {split_name}"

        for _, r in tqdm(df.iterrows(), total=len(df), desc=tqdm_desc):
            serialized_example = _dict_to_example_single(dict(r)).SerializeToString()
            writer.write(serialized_example)

    def Do(
        self,
        input_dict: Dict[str, List[Artifact]],
        output_dict: Dict[str, List[Artifact]],
        exec_properties: Dict[str, Any],
    ) -> None:
        data = self._get_clean_data(
            should_use_local_sample_data=exec_properties[SpecFields.SHOULD_USE_LOCAL_SAMPLE_DATA],
        )

        limit_dataset_size = exec_properties.get(SpecFields.LIMIT_DATASET_SIZE)
        if limit_dataset_size is not None:
            data = data.iloc[:limit_dataset_size]

        train_df, eval_df = self._get_train_eval_split(data)

        size_artifact = self._get_single_artifact(artifacts=output_dict[SpecFields.SIZE])
        self._set_size_artifact(size_artifact=size_artifact, train_df_size=len(train_df), eval_df_size=len(eval_df))

        examples_artifact = self._get_single_artifact(artifacts=output_dict[SpecFields.EXAMPLES])
        self._set_examples_artifact(examples_artifact=examples_artifact)

        self._convert_and_write_tfrecords(
            df=train_df,
            examples_artifact_uri=examples_artifact.uri,
            split_name=TRAIN_SPLIT_NAME,
        )
        self._convert_and_write_tfrecords(
            df=eval_df,
            examples_artifact_uri=examples_artifact.uri,
            split_name=EVAL_SPLIT_NAME,
        )


class SizeArtifact(_TfxArtifact):
    """
    Artifact introduced just as an example. Could be retrieved e.g. using ML Metadata:
    > import ml_metadata as mlmd

    > metadata_connection_config=sqlite_metadata_connection_config(path_to_metadata_db_file)
    > store = mlmd.MetadataStore(metadata_connection_config)
    > store.get_artifacts_by_type(SizeArtifact.TYPE_NAME)
    """

    TYPE_NAME: str = "SizeArtifact"  # type: ignore[assignment]  # In base class defined as `None`
    PROPERTIES = {
        "train": SIZE_PROPERTY,
        "eval": SIZE_PROPERTY,
    }  # type: ignore[assignment]  # In base class defined as `None`


class CustomExampleGenSpec(ComponentSpec):  # type: ignore[no-untyped-call]
    PARAMETERS = {
        SpecFields.SHOULD_USE_LOCAL_SAMPLE_DATA: ExecutionParameter(type=bool),  # type: ignore[no-untyped-call]
        SpecFields.LIMIT_DATASET_SIZE: ExecutionParameter(type=int, optional=True),  # type: ignore[no-untyped-call]
    }
    INPUTS = {}
    OUTPUTS = {
        SpecFields.EXAMPLES: ChannelParameter(type=standard_artifacts.Examples),
        SpecFields.SIZE: ChannelParameter(type=SizeArtifact),
    }


class CustomExampleGen(base_component.BaseComponent):
    SPEC_CLASS = CustomExampleGenSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self, should_use_local_sample_data: bool, limit_dataset_size: Optional[int]) -> None:
        spec = CustomExampleGenSpec(
            **{
                SpecFields.SHOULD_USE_LOCAL_SAMPLE_DATA: should_use_local_sample_data,
                SpecFields.LIMIT_DATASET_SIZE: limit_dataset_size,
                SpecFields.EXAMPLES: Channel(type=standard_artifacts.Examples),
                SpecFields.SIZE: Channel(type=SizeArtifact),
            }
        )  # type: ignore[no-untyped-call]
        super().__init__(spec=spec)
