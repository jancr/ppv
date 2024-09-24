
# core imports
import pathlib
import warnings
import pickle

# 3rd party imports
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

def make_features_numerical(df: pd.DataFrame) -> pd.DataFrame:
    '''Cannot train on boolean values.'''
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    return df


def predict(df, model_dir: str, folds = [0,1,2,3,4]):
    model_dir = pathlib.Path(model_dir)
    exclude_features = [
        ('MS Count', 'start'),
        ('MS Count', 'stop'),
        ('MS Frequency', 'protein_coverage'),
        ('MS Frequency', 'cluster_coverage'),
    ]
    feature_columns = df.columns[ (df.columns.get_level_values(0).str.startswith('MS'))
                                  & ~(df.columns.isin(exclude_features))]
    df = make_features_numerical(df[feature_columns])
    X =  df.values

    all_probs = []
    with warnings.catch_warnings():
        for val in folds:
            for test in folds:
                if val == test:
                    continue
                _path = model_dir / f'model_t{test}_v{val}.pkl'
                model = pickle.load(open(_path, 'rb'))
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                probs = model.predict_proba(X)[:, 1]
                all_probs.append(probs)
    probs = np.stack(all_probs).mean(axis=0)
    return pd.Series(probs, index=df.index, name="PPV")

