import pandas as pd
from pathlib import Path

def load_data(config):
    train_path = Path(config.paths.data.train)
    test_path = Path(config.paths.data.test)

    target_col = config.general.target_col
    id_col = config.general.id_col

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_ids = test_df[id_col].copy()

    X_train = train_df.drop(target_col, axis=1).copy()
    y_train = train_df[target_col].copy()
    X_test = test_df.copy()

    return X_train, y_train, X_test, test_ids