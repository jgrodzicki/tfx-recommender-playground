RECIPE_NAME_FEATURE = "name"
USER_ID_FEATURE = "user_id"

CATEGORICAL_FEATURES = [USER_ID_FEATURE]
NUMERICAL_FEATURES = ["minutes", "n_steps", "n_ingredients"]
TEXT_FEATURES = [RECIPE_NAME_FEATURE]

LABEL_KEY = "rating"

EPOCHS_CONFIG_FIELD_NAME = "epochs"

REQUIRED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES + [LABEL_KEY]


def create_vocab_filename(feature_name: str) -> str:
    return "vocabulary_" + feature_name
