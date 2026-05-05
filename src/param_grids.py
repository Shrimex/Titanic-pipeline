param_grids = {
    "Ridge (L2)": {
        "l1_ratio": [0],
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [5000],
    },
    "Lasso (L1)": {
        "l1_ratio": [1],
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [5000],
    },
    "ElasticNet": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8],
        "max_iter": [5000],
    },

    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "decision_tree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },

    "lightGBM": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [31, 63],
        "subsample": [0.8, 1.0],
    },
    "catboost": {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}