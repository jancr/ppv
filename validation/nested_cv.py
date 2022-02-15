'''
Perform nested cross-validation for all models. 
Save results (performance of each fold) as csv.

If applicable for the model type, we perform grid searches in the inner loop 
to find the best cross-validated set of hyperparameters.

This was written to be run on HPC nodes, so it parallelizes as much as 
possible and starts many parallel processes. (parallize different models, grid search, and inner loop training)

'''

from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Tuple, Callable, Any, List, Optional

# modeling stuff.
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ppv.model import PPVModelVector



def make_features_numerical(df: pd.DataFrame) -> pd.DataFrame:
    '''Cannot train on boolean values.'''
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    return df

def train_eval_random_forest(train_X: pd.DataFrame, 
                             train_y: pd.Series, 
                             valid_X: pd.DataFrame, 
                             valid_y: pd.Series,
                             test_X: pd.DataFrame,
                             test_y: pd.DataFrame,
                             config: Dict[str, Any],
                             ) -> Tuple[np.ndarray, np.ndarray, sklearn.base.ClassifierMixin] :
    
    train_X = make_features_numerical(train_X)
    model = RandomForestClassifier(**config)
    model.fit(train_X.values, train_y.cat.codes)

    valid_probs = model.predict_proba(valid_X.values)
    test_probs =  model.predict_proba(test_X.values)

    return valid_probs[:,1], test_probs[:,1], model


def train_eval_bayes_logreg(train_X: pd.DataFrame, 
                            train_y: pd.Series, 
                            valid_X: pd.DataFrame, 
                            valid_y: pd.Series,
                            test_X: pd.DataFrame,
                            test_y: pd.DataFrame,
                            config: Dict[str, Any],
                            ) -> Tuple[np.ndarray, np.ndarray, PPVModelVector]:

    prior = float(train_y.astype(bool).sum() / train_y.shape[0])
    import pymc3 as pm

    train_X = train_X.copy()
    train_X[('Annotations', 'Known')] = train_y # add y back to dataframe because of PPVModel implementation.
    model =  PPVModelVector(train_X, true_prior = prior)
    
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

def train_eval_freq_logreg_smote(train_X: pd.DataFrame, 
                             train_y: pd.Series, 
                             valid_X: pd.DataFrame, 
                             valid_y: pd.Series,
                             test_X: pd.DataFrame,
                             test_y: pd.DataFrame,
                             config: Dict[str, Any],
                             ) -> Tuple[np.ndarray, Pipeline]:
    
    from imblearn.over_sampling import SMOTE
    train_X = make_features_numerical(train_X)
    train_X, train_y = SMOTE().fit_resample(train_X.values, train_y.cat.codes)

    model = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(**config))])
    model.fit(train_X, train_y)

    valid_probs = model.predict_proba(valid_X.values)
    test_probs =  model.predict_proba(test_X.values)

    return valid_probs[:,1], test_probs[:,1], model

def train_eval_svm(train_X: pd.DataFrame, 
                             train_y: pd.Series, 
                             valid_X: pd.DataFrame, 
                             valid_y: pd.Series,
                             test_X: pd.DataFrame,
                             test_y: pd.DataFrame,
                             config: Dict[str, Any],
                             ) -> Tuple[np.ndarray, Pipeline]:

    model = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC(**config))])
    model.fit(train_X, train_y)

    valid_probs = model.decision_function(valid_X.values)
    test_probs =  model.decision_function(test_X.values)

    return valid_probs[:,1], test_probs[:,1], model


# Define the models we use and the helper functions to learn them.
MODEL_DICT = {
    'SVC': (LinearSVC, train_eval_svm),
    'b_logreg': (PPVModelVector, train_eval_bayes_logreg),
    'f_logreg': (LogisticRegression, train_eval_freq_logreg),
    'f_loreg_smote': (LogisticRegression, train_eval_freq_logreg_smote),
    'RF': (RandomForestClassifier, train_eval_random_forest)
}


def _run_inner_fold(validation_fold: int, 
                    test_fold: int,
                    inner_folds: List[int],
                    df: pd.DataFrame, 
                    test_X: pd.DataFrame,
                    test_y: pd.Series,
                    best_config: Dict[str, Any],
                    model_train_fn: Callable, 
                    feature_columns: List[Tuple[str, str]], 
                    target_column: Tuple[str, str], 
                    model_save_path:str):
    '''Define operations in inner loop as function so that it can be parallelized.'''
    train_folds = [x for x in inner_folds if x != validation_fold]

    train_df = df.loc[df['Annotations']['Fold'].isin(train_folds)] 
    valid_df = df.loc[df['Annotations']['Fold'] == validation_fold]

    train_folds = [x for x in inner_folds if x != validation_fold]

    train_df = df.loc[df['Annotations']['Fold'].isin(train_folds)] 
    valid_df = df.loc[df['Annotations']['Fold'] == validation_fold]

    train_X, train_y = train_df[feature_columns], train_df[target_column]
    valid_X, valid_y = valid_df[feature_columns], valid_df[target_column]

    valid_probs, test_probs, model = model_train_fn(train_X, train_y, valid_X, valid_y, test_X, test_y, best_config)

    valid_auc = roc_auc_score(valid_y.cat.codes, valid_probs)
    test_auc = roc_auc_score(test_y.cat.codes, test_probs)
    
    # save all models with pickle to be library agnostic.
    pickle.dump(model, open(os.path.join(model_save_path, f'model_t{test_fold}_v{validation_fold}.pkl'), 'wb'))
    return valid_auc, test_auc


def nested_cv_loop(
    df: pd.DataFrame, 
    model_name: str, 
    model_save_path: str,
    search_grid: Optional[Dict[str, List[Any]]] = None,
    parallelize_inner: bool = False
    ) -> pd.Series:
    """Outer loop for testing, inner loop for validation. trains n_fold * (n_fold-1) models and evaluates test performances.

    Args:
        df (pd.DataFrame): 
            Full data for training.
        model_save_path (str): Path at which to save all models.

    Returns:
        pd.Series: The test performance of each model.
    """

    if model_name not in MODEL_DICT.keys():
        raise NotImplementedError(model_name)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    
    model_train_fn = MODEL_DICT[model_name][1]

    feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]
    target_column =  ( 'Annotations','Known')


    outer_folds = list(df['Annotations']['Fold'].unique())
    
    results = {}
    # loop and split data.
    for test_fold in outer_folds:

        test_df = df.loc[df['Annotations']['Fold'] == test_fold]
        test_X, test_y = test_df[feature_columns], test_df[target_column]

        # standard cross-validation over the inner folds.
        inner_folds =  [x for x in outer_folds if x != test_fold]
        
        # If we have a grid, first perform grid search for the inner loop folds. Then make all the inner loop models
        # and predict the test set.
        if search_grid is not None:
            best_config, results_table = cv_gridsearch(df, model_name, search_grid, test_fold)
            results_table.to_csv(os.path.join(model_save_path, f'gridsearch_t{test_fold}.csv'))
            print(f'Outer loop {test_fold}: Best config:', best_config)
        else:
            best_config = {}        
    




        jobs = {}
        if parallelize_inner:
            with ProcessPoolExecutor(max_workers=4) as executor:
                for validation_fold in inner_folds:
                    future = executor.submit(_run_inner_fold, validation_fold, test_fold, inner_folds, df, test_X, test_y, best_config, model_train_fn, feature_columns, target_column, model_save_path)
                    jobs[(test_fold, validation_fold)] = future

                for k, future in jobs.items():
                    valid_auc, test_auc = future.result()
                    print(model_name, test_fold, validation_fold, valid_auc)
                    results[k] = test_auc 


        else:
            for validation_fold in inner_folds:

                valid_auc, test_auc = _run_inner_fold(validation_fold, test_fold, inner_folds, df, test_X, test_y, best_config, model_train_fn, feature_columns, target_column, model_save_path)
                print(model_name, test_fold, validation_fold, valid_auc)
                results[(test_fold, validation_fold)] = test_auc            

            


    results = pd.Series(results)
    results.name = model_name
    pd.DataFrame(results).to_csv(os.path.join(model_save_path, 'test_performances.csv'))
    return results



def cv_gridsearch(
    df: pd.DataFrame, 
    model_name: str,
    grid: Dict[str, List[Any]],
    test_fold: int,
    ) -> Dict[str, Any]:

    feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]
    target_column =  ( 'Annotations','Known')

    folds = list(df['Annotations']['Fold'].unique())
    # GridSearchCV needs an iterable of (train_idx, valid_idx) arrays
    inner_folds =  [x for x in folds if x != test_fold]
    train_df = df.loc[df['Annotations']['Fold'].isin(inner_folds)]
    cv_idx = [(np.where(train_df[('Annotations','Fold')] != v)[0], np.where(train_df[('Annotations','Fold')]==v)[0] ) for v in inner_folds]

    # Set up model and search.
    if model_name in ['SVC', 'b_logreg', 'f_logreg']:
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', MODEL_DICT[model_name][0]())
            ])
    elif model_name == 'f_logreg_smote':
        #https://towardsdatascience.com/the-right-way-of-using-smote-with-cross-validation-92a8d09d00c7
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        model = ImbPipeline([
            ('smote', SMOTE()), 
            ('scaler', StandardScaler()), 
            ('clf', MODEL_DICT[model_name][0]())
            ])
    else:
        model = MODEL_DICT[model_name][0]()

    search = GridSearchCV(model, grid, cv=cv_idx, refit=False, scoring='roc_auc', n_jobs=30, verbose=1)

    # Set up train data and run search.
    train_X, train_y = train_df[feature_columns], train_df[target_column]
    train_df = make_features_numerical(train_df)
    search.fit(train_X.values.copy(), train_y.cat.codes.values.copy())

    best_params = {k.removeprefix('clf__'):v for k,v in search.best_params_.items()}
    return best_params, pd.DataFrame.from_dict(search.cv_results_)




def main():
    df =  pd.read_pickle('mouse_features_paper.pickle')

    #nested_cv_loop(df, 'SVC', 'debug/cv_svc', {'clf__C': [1, 10, 100], 'clf__max_iter':[2000]} )

    runs = [
        (df, 'b_logreg','debug/cv_bayes_logreg/', None, True),
        (df, 'RF', 'debug/cv_rf/', {'n_estimators':[30,50,70,90,100,120, 140], 'max_depth':[3,5,10, 100]}, True),
        (df, 'f_logreg', 'debug/cv_f_logreg/', None, True),
        (df, 'f_logreg_smote', 'debug/cv_f_logreg_smote/', None, True),
        (df, 'SVC', 'debug/cv_svc', {'clf__C': [1, 10, 100], 'clf__max_iter':[3000]} )
    ]

    jobs = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for run_params in runs:
            future = executor.submit(nested_cv_loop, *run_params)
            jobs.append(future)


    results = [job.result() for job in jobs]
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv('nested_crossvalidation_result.csv')
    

if __name__ == '__main__':
        
    main()
    
