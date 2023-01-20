import os.path
from typing import Any, Dict

import tensorflow as tf
import tensorflow_transform as tft

from src.components.consts import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    TEXT_FEATURES,
    VOCABULARY_FILE_PREFIX,
)

MODULE_FILE = os.path.abspath(__file__)


def preprocessing_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    outputs = {}
    for feature in CATEGORICAL_FEATURES:
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            vocab_filename=VOCABULARY_FILE_PREFIX + feature,
            num_oov_buckets=1,
        )

    for feature in NUMERICAL_FEATURES:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])

    for feature in TEXT_FEATURES:
        preprocessed = tf.strings.regex_replace(inputs[feature], " +", " ")
        outputs[feature] = preprocessed

        # Vocabulary will be needed later on during the training
        tft.compute_and_apply_vocabulary(preprocessed, vocab_filename="vocabulary_" + feature)

    outputs[LABEL_KEY] = tft.scale_to_z_score(inputs[LABEL_KEY])

    return outputs
