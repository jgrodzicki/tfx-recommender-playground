import logging
from typing import Dict

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow_transform import TFTransformOutput

from src.components.common import RECIPE_NAME_FEATURE, USER_ID_FEATURE, create_vocab_filename, get_logger


class RecommenderModel(tfrs.Model):  # type: ignore[misc]  # Class cannot subclass "Model" (has type "Any")
    def __init__(
        self,
        recipe_embedder: tf.keras.Model,
        user_embedder: tf.keras.Model,
        retrieval_task: tfrs.tasks.Retrieval,
    ):
        super().__init__()
        self.recipe_embedder = recipe_embedder
        self.user_embedder = user_embedder
        self.retrieval_task = retrieval_task

    def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        recipe_embeddings = self.recipe_embedder(features[RECIPE_NAME_FEATURE])
        user_embeddings = self.user_embedder(features[USER_ID_FEATURE])

        user_embeddings = tf.squeeze(user_embeddings, axis=1)

        should_compute_metrics = not training  # Do not compute metrics when training to improve the performance
        task: tf.Tensor = self.retrieval_task(
            query_embeddings=user_embeddings,
            candidate_embeddings=recipe_embeddings,
            compute_metrics=should_compute_metrics,
        )
        return task


class RecommenderModelFactory:
    EMBEDDING_DIMENSION = 32

    @classmethod
    def create(
        cls,
        tf_transform_output: TFTransformOutput,
        recipe_dataset: tf.data.Dataset,
    ) -> RecommenderModel:
        logger = get_logger(f"{__name__}:{cls.__name__}")
        logger.info("Creating recommender model")

        recipe_embedder = RecommenderModelFactory._build_recipe_embedder(
            tf_transform_output=tf_transform_output,
            logger=logger,
        )
        user_embedder = RecommenderModelFactory._build_user_embedder(
            tf_transform_output=tf_transform_output,
            logger=logger,
        )

        metrics = tfrs.metrics.FactorizedTopK(candidates=recipe_dataset.batch(128).map(recipe_embedder))
        retrieval_task = tfrs.tasks.Retrieval(metrics=metrics)

        return RecommenderModel(
            recipe_embedder=recipe_embedder,
            user_embedder=user_embedder,
            retrieval_task=retrieval_task,
        )

    @staticmethod
    def _build_user_embedder(tf_transform_output: TFTransformOutput, logger: logging.Logger) -> tf.keras.Model:
        logger.info("Building user id embedder")

        unique_ids = tf_transform_output.vocabulary_by_name(vocab_filename=create_vocab_filename(USER_ID_FEATURE))
        unique_ids_str = [b.decode() for b in unique_ids]
        logger.info(f"Found {len(unique_ids_str)} unique users to build the embedding")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=len(unique_ids_str) + 1,
                    output_dim=RecommenderModelFactory.EMBEDDING_DIMENSION,
                    name=f"{USER_ID_FEATURE}_embedding",
                ),
            ]
        )
        return model

    @staticmethod
    def _build_recipe_embedder(tf_transform_output: TFTransformOutput, logger: logging.Logger) -> tf.keras.Model:
        logger.info("Building recipe embedder")
        unique_ids = tf_transform_output.vocabulary_by_name(vocab_filename=create_vocab_filename(RECIPE_NAME_FEATURE))
        logger.info(f"Found {len(unique_ids)} unique recipes to build the embedding")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=unique_ids, name=f"{RECIPE_NAME_FEATURE}_lookup"),
                tf.keras.layers.Embedding(
                    input_dim=len(unique_ids) + 1,
                    output_dim=RecommenderModelFactory.EMBEDDING_DIMENSION,
                    name=f"{RECIPE_NAME_FEATURE}_embedding",
                ),
            ]
        )
        return model
