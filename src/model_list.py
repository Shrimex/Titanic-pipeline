from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.neural_network import TitanicNN


def get_models(random_state: int = 42) -> dict:
    '''Список моделей для решения задачи классификации'''
    models = {
        "dummy_classifier": DummyClassifier(strategy='most_frequent'),

        'Ridge (L2)': LogisticRegression( l1_ratio=0.0, solver='lbfgs',     C=1.0, max_iter=5000),
        'Lasso (L1)': LogisticRegression( l1_ratio=1.0, solver='saga', C=1.0, max_iter=5000),
        'ElasticNet': LogisticRegression( l1_ratio=0.5, solver='saga',  C=1.0, max_iter=5000),

        'KNN':           KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state),

        "lightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbose=-1),
        "catboost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=random_state),
        "XGBoost":  XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state,
                                   use_label_encoder=False, eval_metric='logloss', verbosity = 0),

        "NeuralNet_basic": TitanicNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
        ),
        "NeuralNet_sgd": TitanicNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            optimizer='sgd'
        ),
        "NeuralNet_rmsprop": TitanicNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            optimizer='rmsprop'
        ),
        "NeuralNet_cosine": TitanicNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            scheduler='cosine',
            scheduler_kwargs={"T_max": 50},
        ),
        "NeuralNet_rop": TitanicNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            scheduler='reduce_on_plateau',
            scheduler_kwargs={
                "mode": "min",
                "factor": 0.5,
                "patience": 5
            },
        ),
        "NeuralNet_combination": TitanicNN(
            hidden_dims=[64, 32],
            dropout=0.3,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=15,
            random_state=random_state,
            optimizer='adam',
            scheduler='reduce_on_plateau',
            scheduler_kwargs={
                "mode": "min",
                "factor": 0.5,
                "patience": 5
            },
        ),
        "voting": VotingClassifier(
            estimators=[
                ('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbose=-1)),
                ('cat', CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=random_state)),
                ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbosity=0)),
            ],
            voting='soft',
        ),

        "stacking_ridge": StackingClassifier(
            estimators=[
                ('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbose=-1)),
                ('cat', CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=random_state)),
                ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbosity=0)),
                ('rf',  RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)),
            ],
            final_estimator=LogisticRegression(),
        ),

        "stacking_lasso": StackingClassifier(
            estimators=[
                ('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbose=-1)),
                ('cat', CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=random_state)),
                ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state, verbosity=0)),
                ('rf',  RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)),
            ],
            final_estimator=LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=5000),
            cv=5,
        ),
    }
    return models


def get_model_by_name(model_name: str, random_state: int = 42):
    models = get_models(random_state=random_state)

    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Неизвестная модель '{model_name}'. Доступные: {available}")

    return models[model_name]