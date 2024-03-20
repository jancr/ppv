# PPV - training and evaluation codebase

### 1. Splitting the data for nested cross-validation

The code assumes that the feature generation pipeline was run successfully, transforming the peptidomics data into a pandas dataframe stored as `mouse_features_paper.pickle`. To split the data into 5 folds, run
```
python3 make_crossvalidation_split.py mouse_features_paper.pickle mouse_features_paper_sklearn.pickle
python3 make_crossvalidation_split.py mouse_features_paper.pickle mouse_features_paper_assembly_sklearn.pickle --use_all
```
### 2. Training the ppv model

The script `nested_cv.py` trains our ML models in nested cross-validation, yielding 20 models. The script also trains various baseline ML models. Internally, the PPV model is called `f_logreg` (=frequentist logistic regression). If you want to skip training baseline ML models, comment out the respective models in `runs` starting from line 381. 
```
python3 nested_cv.py -d mouse_features_paper_sklearn.pickle -od nested_cv
```
This creates a directory called `nested_cv` that contains the cross-validated models.

### 3. Evaluation 

The jupyter notebook `manuscript_figures.ipynb` produces the performance plots shown in the manuscript from `nested_cv` and the saved `mouse_features_paper_sklearn.pickle` feature data.


### 4. Making new predictions

The full PPV model is an ensemble of the cross-validated models. To apply it to new data, use the following code snippet that takes care of averaging the 20 predictions.

```
def make_features_numerical(df: pd.DataFrame) -> pd.DataFrame:
    '''Cannot train on boolean values.'''
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    return df

def predict_probabilities(df, model_dir: str = "nested_cv/cv_f_logreg", folds = [0,1,2,3,4]):

    df_X = make_features_numerical(df_X)
    X =  df_X.values

    all_probs = []
    # predict from all the test models and average probabilities.
    for val in folds:
        for test in folds:
            if val == test:
                continue

            model = pickle.load(open(os.path.join(results_dir, f'model_t{test}_v{val}.pkl'), 'rb'))
            probs = model.predict_proba(X)[:, 1]
            all_probs.append(probs)

    probs = np.stack(all_probs).mean(axis=0)
    return probs

exclude_features = [
    (    'MS Count',             'start'),
    (    'MS Count',             'stop'),
    (    'MS Frequency',        'protein_coverage'),
    (    'MS Frequency',        'cluster_coverage'),
]
feature_columns = df.columns[ (df.columns.get_level_values(0).str.startswith('MS')) & ~(df.columns.isin(exclude_features))]    

df['Annotations', 'PPV'] = predict_probabilities(df[feature_columns])
```