import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['Initial'] = df.Name.str.extract(r'([A-Za-z]+)\.')
    df['Initial'] = df['Initial'].replace({
        'Mlle': 'Miss',
        'Mme':  'Miss',
        'Ms':   'Miss',
        'Dr':   'Mr',
        'Major':'Mr',
        'Lady': 'Mrs',
        'Countess': 'Mrs',
        'Capt': 'Mr',
        'Sir':  'Mr',
        'Don':  'Mr',
    })
    df.loc[~df['Initial'].isin(['Mr', 'Miss', 'Mrs']), 'Initial'] = 'Other'
    df['Family_Size'] = df['Parch'] + df['SibSp']
    df['Alone'] = np.where(df['Family_Size'] == 0, 1, 0)

    df['Family_Type'] = pd.cut(
        df['Family_Size'],
        bins=[-1, 0, 3, 6, 20],
        labels=[0, 1, 2, 3]
    ).astype(int)
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('U')

    return df