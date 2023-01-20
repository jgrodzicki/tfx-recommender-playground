import os
from typing import List, Union

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow_transform import TFTransformOutput
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.public import tfxio

from src.components.common import create_vocab_filename
from src.components.consts import EPOCHS_CONFIG_FIELD_NAME, RECIPE_NAME_FEATURE
from src.model import RecommenderModelFactory

MODULE_FILE = os.path.abspath(__file__)
FeatureSpecType = Union[tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature, tf.io.RaggedFeature, tf.io.SparseFeature]


TRAIN_BATCH_SIZE = 100
EVAL_BATCH_SIZE = 50


def _create_dataset(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: TFTransformOutput,
    batch_size: int,
) -> tf.data.Dataset:
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        tf_transform_output.transformed_metadata.schema,
    )
    return dataset


def _get_serve_tf_examples_fn(
    model: tf.keras.Model,
    tf_transform_output: TFTransformOutput,
) -> tf.types.experimental.GenericFunction:
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples: tf.string) -> tf.Tensor:
        feature_spec = tf_transform_output.raw_feature_spec()
        user_id_spec = feature_spec["user_id"]

        parsed_features = tf.io.parse_example(serialized_tf_examples, {"user_id": user_id_spec})

        transformed_features = model.tft_layer(parsed_features)

        model_input = {"user_id_embedding_input": transformed_features["user_id"]}
        result: tf.Tensor = model(model_input)

        return result

    return serve_tf_examples_fn


def _create_recipe_dataset(tf_transform_output: TFTransformOutput) -> tf.data.Dataset:
    vocabulary = tf_transform_output.vocabulary_by_name(create_vocab_filename(feature_name=RECIPE_NAME_FEATURE))
    return tf.data.Dataset.from_tensor_slices(vocabulary)


def run_fn(fn_args: FnArgs) -> None:
    tf_transform_output = TFTransformOutput(fn_args.transform_output)

    train_dataset = _create_dataset(
        file_pattern=fn_args.train_files,
        data_accessor=fn_args.data_accessor,
        tf_transform_output=tf_transform_output,
        batch_size=TRAIN_BATCH_SIZE,
    )
    eval_dataset = _create_dataset(
        file_pattern=fn_args.eval_files,
        data_accessor=fn_args.data_accessor,
        tf_transform_output=tf_transform_output,
        batch_size=EVAL_BATCH_SIZE,
    )

    recipe_dataset = _create_recipe_dataset(tf_transform_output=tf_transform_output)

    model = RecommenderModelFactory.create(
        tf_transform_output=tf_transform_output,
        recipe_dataset=recipe_dataset,
    )

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq="batch")

    model.fit(
        x=train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=fn_args.custom_config[EPOCHS_CONFIG_FIELD_NAME],
        callbacks=[tensorboard_callback],
    )

    index = tfrs.layers.factorized_top_k.BruteForce(model.user_embedder)

    index.index_from_dataset(
        tf.data.Dataset.zip(
            (
                recipe_dataset.batch(100),
                recipe_dataset.batch(100).map(model.recipe_embedder),
            )
        )
    )
    # Run once so that we can get the right signatures into SavedModel
    _ = index(tf.constant([42]))

    serving_func = _get_serve_tf_examples_fn(index, tf_transform_output).get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    )
    signatures = {"serving_default": serving_func}
    index.save(fn_args.serving_model_dir, signatures=signatures)
