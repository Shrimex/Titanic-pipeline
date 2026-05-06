import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score



def evaluate_models(
    
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    '''Функция для вычисления метрик моделей'''
    results = []

    for model_name, model in models.items():
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        f1_scores = cross_val_score(model, X, y, scoring='f1', cv=cv)

        results.append({
            "model_name": model_name,
            "cv_accuracy_mean": accuracy_scores.mean(),
            "cv_accuracy_std": accuracy_scores.std(),
            "cv_f1_mean": f1_scores.mean(),
            "cv_f1_std": f1_scores.std(),
        })

    result_df = (
        pd.DataFrame(results)
        .sort_values("cv_accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )
    return result_df