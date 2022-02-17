import argparse
import pickle
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pymc3 as pm

from ppv.model import PPVModelPaper

def make_features_numerical(df: pd.DataFrame) -> pd.DataFrame:
    '''Cannot train on boolean values.'''
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    return df

def train_eval_bayes_logreg(train_X: pd.DataFrame, 
                            train_y: pd.Series, 
                            valid_X: pd.DataFrame, 
                            valid_y: pd.Series,
                            test_X: pd.DataFrame,
                            test_y: pd.DataFrame,
                            ) -> Tuple[np.ndarray, np.ndarray, PPVModelPaper]:

    prior = float(train_y.astype(bool).sum() / train_y.shape[0])

    train_X = train_X.copy()
    train_X[('Annotations', 'Known')] = train_y # add y back to dataframe because of PPVModel implementation.
    model =  PPVModelPaper(train_X, true_prior = prior)
    
    with model.model:
        trace = pm.sample(2000, cores=7, chains=7, target_accept=0.9, return_inferencedata=True)

    model.add_trace(trace, true_prior = prior)

    valid_probs = model.predict(valid_X)
    test_probs = model.predict(test_X)
    return valid_probs, test_probs, model

def train_eval_freq_logreg(train_X: pd.DataFrame, 
                             train_y: pd.Series, 
                             valid_X: pd.DataFrame, 
                             valid_y: pd.Series,
                             test_X: pd.DataFrame,
                             test_y: pd.DataFrame,
                             config: Dict[str, Any],
                             ) -> Tuple[np.ndarray, Pipeline]:

    train_X = make_features_numerical(train_X)
    model = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(**config, max_iter=1000))])
    model.fit(train_X.values, train_y.cat.codes)

    valid_probs = model.predict_proba(valid_X.values)
    test_probs =  model.predict_proba(test_X.values)

    return valid_probs[:,1], test_probs[:,1], model


def _run_inner_fold(validation_fold: int, 
                    test_fold: int,
                    inner_folds: List[int],
                    df: pd.DataFrame, 
                    test_X: pd.DataFrame,
                    test_y: pd.Series,
                    feature_columns: List[Tuple[str, str]], 
                    target_column: Tuple[str, str], 
                    model_save_path:str,
                    bayesian: bool = False
                    ):
    '''Define operations in inner loop as function so that it can be parallelized.'''
    train_folds = [x for x in inner_folds if x != validation_fold]

    train_df = df.loc[df['Annotations']['Fold'].isin(train_folds)] 
    valid_df = df.loc[df['Annotations']['Fold'] == validation_fold]

    train_folds = [x for x in inner_folds if x != validation_fold]

    train_df = df.loc[df['Annotations']['Fold'].isin(train_folds)] 
    valid_df = df.loc[df['Annotations']['Fold'] == validation_fold]

    train_X, train_y = train_df[feature_columns], train_df[target_column]
    valid_X, valid_y = valid_df[feature_columns], valid_df[target_column]

    if bayesian:
        valid_probs, test_probs, model = train_eval_bayes_logreg(train_X, train_y, valid_X, valid_y, test_X, test_y)
    else: 
        valid_probs, test_probs, model = train_eval_freq_logreg(train_X, train_y, valid_X, valid_y, test_X, test_y, {})

    valid_auc = roc_auc_score(valid_y.cat.codes, valid_probs)
    test_auc = roc_auc_score(test_y.cat.codes, test_probs)
    
    # save all models with pickle to be library agnostic.
    pickle.dump(model, open(os.path.join(model_save_path, f'model_t{test_fold}_v{validation_fold}.pkl'), 'wb'))
    return valid_auc, test_auc


def nested_cv_loop(
    df: pd.DataFrame, 
    model_save_path: str,
    features_to_use: List[Tuple[str, str]]
    ) -> pd.Series:


    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    
    # TODO select features.
    target_column =  ( 'Annotations','Known')


    outer_folds = list(df['Annotations']['Fold'].unique())
    
    results = {}
    jobs = {}
    # loop and split data.
    with ProcessPoolExecutor(max_workers=9) as executor:
        for test_fold in outer_folds:

            test_df = df.loc[df['Annotations']['Fold'] == test_fold]
            test_X, test_y = test_df[features_to_use], test_df[target_column]

            # standard cross-validation over the inner folds.
            inner_folds =  [x for x in outer_folds if x != test_fold]
        
            for validation_fold in inner_folds:
                future = executor.submit(_run_inner_fold, validation_fold, test_fold, inner_folds, df, test_X, test_y, features_to_use, target_column, model_save_path)
                jobs[(test_fold, validation_fold)] = future

        for k, future in jobs.items():
            valid_auc, test_auc = future.result()
            results[k] = test_auc 


    results = pd.Series(results)
    results.name = 'chemical'
    pd.DataFrame(results).to_csv(os.path.join(model_save_path, 'test_performances.csv'))
    return results

            

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default = 'mouse_features_paper.pickle')
    parser.add_argument('--out_dir', '-od', default='nested_cv')
    args = parser.parse_args()


    df =  pd.read_pickle(args.data)

    # full model (now MS only)
    #feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]

    # chemical model
    feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('Chemical')]
    nested_cv_loop(df, os.path.join(args.out_dir, 'cv_chemical/'), feature_columns)

    # NOTE we do not need to actually train this when using AUC or any other rank-based metric.
    # null model
    #feature_columns = [('Annotations', 'Intensity')]

if __name__ == '__main__':
    main()