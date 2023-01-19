import os
from pathlib import Path
from typing import Any, Dict, List

import kaggle
import pandas as pd
import tensorflow as tf
from tfx.dsl.components.base import base_component, base_executor, executor_spec
from tfx.types import artifact_utils, standard_artifacts
from tfx.types.artifact import Artifact, Property, PropertyType
from tfx.types.channel import Channel
from tfx.types.component_spec import ChannelParameter, ComponentSpec
from tfx.types.standard_artifacts import _TfxArtifact
from tqdm import tqdm

KAGGLE_DATASET = "shuyangli94/food-com-recipes-and-user-interactions"
RAW_RECIPES_FILE_PATH = Path("RAW_recipes.csv")
RAW_INTERACTIONS_FILE_PATH = Path("RAW_interactions.csv")

DATA_FOLDER = Path("data")

SIZE_PROPERTY = Property(type=PropertyType.INT)  # type: ignore[no-untyped-call]


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


def convert_and_write_tfrecords(df: pd.DataFrame, examples_artifact_uri: str, split_name: str) -> None:
    output_folder = f"{examples_artifact_uri}/Split-{split_name}"
    os.makedirs(output_folder)

    writer = tf.io.TFRecordWriter(
        f"{output_folder}/tfrecord.gz",
        options=tf.compat.v1.io.TFRecordCompressionType.GZIP,
    )

    for _, r in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Converting to TFRecord for split - {split_name}",
    ):
        serialized_example = _dict_to_example_single(dict(r)).SerializeToString()
        writer.write(serialized_example)


class Executor(base_executor.BaseExecutor):
    def Do(
        self,
        input_dict: Dict[str, List[Artifact]],
        output_dict: Dict[str, List[Artifact]],
        exec_properties: Dict[str, Any],
    ) -> None:
        for filename in [RAW_RECIPES_FILE_PATH, RAW_INTERACTIONS_FILE_PATH]:
            kaggle.api.dataset_download_file(
                dataset=KAGGLE_DATASET,
                file_name=str(filename),
                path=str(DATA_FOLDER),
            )

        recipes_df = pd.read_csv(str(DATA_FOLDER / RAW_RECIPES_FILE_PATH) + ".zip")
        interactions_df = pd.read_csv(str(DATA_FOLDER / RAW_INTERACTIONS_FILE_PATH) + ".zip")

        merged_df_full = recipes_df.merge(
            right=interactions_df,
            left_on="id",
            right_on="recipe_id",
        )
        required_columns = ["user_id", "recipe_id", "rating", "minutes", "n_steps", "n_ingredients", "name"]
        merged_df = merged_df_full[required_columns]
        merged_df = merged_df.dropna(how="any")

        train_df = merged_df.sample(frac=0.7)
        eval_df = merged_df.drop(train_df.index)

        size_artifact = artifact_utils.get_single_instance(output_dict["size"])
        size_artifact.train = len(train_df)
        size_artifact.eval = len(eval_df)

        examples_artifact_uri = artifact_utils.get_single_uri(output_dict["examples"])

        artifact = artifact_utils.get_single_instance(output_dict["examples"])
        artifact.split_names = artifact_utils.encode_split_names(["train", "eval"])

        convert_and_write_tfrecords(
            df=train_df,
            examples_artifact_uri=examples_artifact_uri,
            split_name="train",
        )
        convert_and_write_tfrecords(
            df=eval_df,
            examples_artifact_uri=examples_artifact_uri,
            split_name="eval",
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
    PARAMETERS = {}
    INPUTS = {}
    OUTPUTS = {
        "examples": ChannelParameter(type=standard_artifacts.Examples),
        "size": ChannelParameter(type=SizeArtifact),
    }


class CustomExampleGen(base_component.BaseComponent):
    SPEC_CLASS = CustomExampleGenSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self) -> None:
        spec = CustomExampleGenSpec(
            examples=Channel(type=standard_artifacts.Examples),
            size=Channel(type=SizeArtifact),
        )  # type: ignore[no-untyped-call]
        super().__init__(spec=spec)
