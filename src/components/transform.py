import os.path
from typing import Any, Dict

import tensorflow as tf
import tensorflow_transform as tft

from src.components.common import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    TEXT_FEATURES,
    create_vocab_filename,
    get_logger,
)

MODULE_FILE = os.path.abspath(__file__)


def preprocessing_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_logger(__name__)
    logger.info("Preprocessing data")

    outputs = {}
    for feature in CATEGORICAL_FEATURES:
        logger.info(f"Computing and applying vocabulary for the feature: {feature}")
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            vocab_filename=create_vocab_filename(feature_name=feature),
            num_oov_buckets=1,
        )

    for feature in NUMERICAL_FEATURES:
        logger.info(f"Scaling to z_score feature: {feature}")
        outputs[feature] = tft.scale_to_z_score(inputs[feature])

    for feature in TEXT_FEATURES:
        logger.info(f"Cleaning and computing and applying vocabulary for the feature: {feature}")
        preprocessed = tf.strings.regex_replace(inputs[feature], " +", " ")
        outputs[feature] = preprocessed

        # Vocabulary will be needed later on during the training
        tft.compute_and_apply_vocabulary(preprocessed, vocab_filename=create_vocab_filename(feature_name=feature))

    logger.info(f"Scaling to z_score {LABEL_KEY}")
    outputs[LABEL_KEY] = tft.scale_to_z_score(inputs[LABEL_KEY])

    return outputs
