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
)

MODULE_FILE = os.path.abspath(__file__)


def preprocessing_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    outputs = {}
    for feature in CATEGORICAL_FEATURES:
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            vocab_filename=create_vocab_filename(feature_name=feature),
            num_oov_buckets=1,
        )

    for feature in NUMERICAL_FEATURES:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])

    for feature in TEXT_FEATURES:
        preprocessed = tf.strings.regex_replace(inputs[feature], " +", " ")
        outputs[feature] = preprocessed

        # Vocabulary will be needed later on during the training
        tft.compute_and_apply_vocabulary(preprocessed, vocab_filename=create_vocab_filename(feature_name=feature))

    outputs[LABEL_KEY] = tft.scale_to_z_score(inputs[LABEL_KEY])

    return outputs
