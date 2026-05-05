from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.data import load_data
from src.evaluation import evaluate_models
from src.model_list import get_models
from src.preprocessing import preprocess_features
from src.param_grids import param_grids
from src.utils import ensure_dir, save_dataframe, save_json
from sklearn.model_selection import StratifiedKFold

def run_training_pipeline(config):
    metrics_dir     = Path(config.paths.outputs.metrics)
    models_dir      = Path(config.paths.outputs.models)
    submissions_dir = Path(config.paths.outputs.submissions)

    ensure_dir(metrics_dir)
    ensure_dir(models_dir)
    ensure_dir(submissions_dir)

    X_train, y_train, X_test, test_ids = load_data(
        config
    )
    X_train_processed, X_test_processed = preprocess_features(
        X_train=X_train,
        X_test=X_test,
        id_col=config.general.id_col,
    )

    models = get_models(random_state=config.general.seed)
    results_df = evaluate_models(
        models=models,
        X=X_train_processed,
        y=y_train,
        n_folds=config.cross_validation.n_folds,
        random_state=config.general.seed,
    )

    best_model_name  = results_df.iloc[0]["model_name"]
    best_cv_f1       = float(results_df.iloc[0]["cv_f1_mean"])
    best_cv_accuracy = float(results_df.iloc[0]["cv_accuracy_mean"])

    print("Результаты кросс-валидации (дефолтные параметры):")
    print(results_df.to_string(index=False))
    print(f"\nЛучшая модель: {best_model_name} (f1={best_cv_f1:.4f})")

    grid_results = []
    cv=StratifiedKFold(
                n_splits=config.cross_validation.n_folds,
                shuffle=True,
                random_state=config.general.seed
            )

    for model_name, model in models.items():
        if model_name == "dummy_classifier" or model_name not in param_grids:
            continue

        print(f"GridSearchCV для '{model_name}'...")
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="f1",
            n_jobs=6,
            refit=True,
        )
        search.fit(X_train_processed, y_train)

        grid_results.append({
            "model_name": model_name,
            "best_f1": search.best_score_,
            "best_params": search.best_params_,
            "best_estimator": search.best_estimator_,
        })
        print(f"  f1={search.best_score_:.4f} params={search.best_params_}")

    grid_results_df = (
        pd.DataFrame(grid_results)
        .sort_values("best_f1", ascending=False)
        .reset_index(drop=True)
    )

    best = grid_results_df.iloc[0]
    final_model      = best["best_estimator"]
    selected_model_name = best["model_name"]
    best_params      = best["best_params"]
    best_search_f1   = best["best_f1"]

    print("\nРезультаты после GridSearchCV:")
    print(grid_results_df[["model_name", "best_f1"]].to_string(index=False))

    test_predictions = final_model.predict(X_test_processed)
    submission_df = pd.DataFrame({
        config.general.id_col:     test_ids,
        config.general.target_col: test_predictions,
    })

    if config.saving.save_metrics:
        save_dataframe(results_df, metrics_dir / "cv_results.csv")
        save_json(
            {
                "best_cv_model":        best_model_name,
                "best_cv_f1":           best_cv_f1,
                "best_cv_accuracy":     best_cv_accuracy,
                "selected_final_model": selected_model_name,
                "best_params":          best_params,
                "best_search_f1":       best_search_f1,
            },
            metrics_dir / "summary.json",
        )

    if config.saving.save_submission:
        save_dataframe(
            submission_df,
            submissions_dir / f"submission_{selected_model_name}.csv",
        )

    if config.saving.save_model:
        joblib.dump(
            final_model,
            models_dir / f"{selected_model_name}_model.joblib",
        )

    print("\nОбучение завершено.")
    print(f"Выбранная модель:      {selected_model_name}")
    print(f"F1 лучшей конфигурации: {best_search_f1:.4f}")
    print(f"Лучшие параметры:      {best_params}")