from omegaconf import OmegaConf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

config={
    'general':{
        'seed':142,
        'target_col': 'Survived',
        'id_col':'PassengerId',
        'run_name':'baseline_v1'
    },
    'paths':{
        'root':str(PROJECT_ROOT),
        'data':{
            'train':"${paths.root}/Data/train.csv",
            'test':"${paths.root}/Data/test.csv",
        },
        'outputs':{
            'submissions': "${paths.root}/outputs/submissions",
            'metrics': "${paths.root}/outputs/metrics",
            'models': "${paths.root}/outputs/models",
        }
    },
    'cross_validation':{
        'n_folds':5,
    },
    'model':{
        'default_model':'CatBoost',
        'use_best_cv':True,
    },
    'saving':{
        'save_metrics':True,
        'save_model':True,
        'save_submission':True,
    },
    'nn':{
        'hidden_dims': [128, 64, 32],
        'dropout_rate':0.3,
        'lr':1e-3,
        'num_epoch':200,
        'batch_size':32,
        'patience':15,
    },
}
config=OmegaConf.create(config)