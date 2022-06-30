```python
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
```


```python
results_dir = 'nested_cv'
```


```python
# make a barplot to compare model AUCs.

results = []
for cv_run in os.listdir(results_dir):
    try:
        metrics = pd.read_csv(os.path.join(results_dir, cv_run, 'test_performances.csv'), index_col=[0,1])
        results.append(metrics)
    except FileNotFoundError:
        pass

test_performances = results[0].join(results[1:])
```


```python
test_performances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>SVC</th>
      <th>RF</th>
      <th>f_logreg</th>
      <th>f_logreg_smote</th>
      <th>chemical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">1</th>
      <th>4</th>
      <td>0.864660</td>
      <td>0.830662</td>
      <td>0.864143</td>
      <td>0.827393</td>
      <td>0.603144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.870188</td>
      <td>0.854480</td>
      <td>0.871651</td>
      <td>0.840151</td>
      <td>0.613644</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.844233</td>
      <td>0.873784</td>
      <td>0.859043</td>
      <td>0.815711</td>
      <td>0.635962</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.869558</td>
      <td>0.834556</td>
      <td>0.867262</td>
      <td>0.829310</td>
      <td>0.590754</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">4</th>
      <th>1</th>
      <td>0.904921</td>
      <td>0.961269</td>
      <td>0.944526</td>
      <td>0.927391</td>
      <td>0.680253</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.860610</td>
      <td>0.954884</td>
      <td>0.931655</td>
      <td>0.894903</td>
      <td>0.713497</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.917797</td>
      <td>0.964887</td>
      <td>0.946775</td>
      <td>0.943707</td>
      <td>0.696543</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.891392</td>
      <td>0.950796</td>
      <td>0.944838</td>
      <td>0.930685</td>
      <td>0.726203</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2</th>
      <th>1</th>
      <td>0.820877</td>
      <td>0.858079</td>
      <td>0.818032</td>
      <td>0.811300</td>
      <td>0.651992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.804997</td>
      <td>0.855658</td>
      <td>0.808415</td>
      <td>0.808155</td>
      <td>0.664748</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.819899</td>
      <td>0.861364</td>
      <td>0.823140</td>
      <td>0.818300</td>
      <td>0.720336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.801818</td>
      <td>0.840868</td>
      <td>0.815611</td>
      <td>0.818399</td>
      <td>0.660478</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">0</th>
      <th>1</th>
      <td>0.888382</td>
      <td>0.871794</td>
      <td>0.897408</td>
      <td>0.884776</td>
      <td>0.544046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.897191</td>
      <td>0.882410</td>
      <td>0.903652</td>
      <td>0.900527</td>
      <td>0.441085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.883893</td>
      <td>0.885074</td>
      <td>0.893778</td>
      <td>0.907413</td>
      <td>0.544761</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.878407</td>
      <td>0.880492</td>
      <td>0.887349</td>
      <td>0.886020</td>
      <td>0.505062</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">3</th>
      <th>1</th>
      <td>0.881405</td>
      <td>0.861215</td>
      <td>0.890555</td>
      <td>0.886002</td>
      <td>0.635835</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.884499</td>
      <td>0.858587</td>
      <td>0.886942</td>
      <td>0.891737</td>
      <td>0.711198</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.868893</td>
      <td>0.858393</td>
      <td>0.879404</td>
      <td>0.901151</td>
      <td>0.661713</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.871636</td>
      <td>0.872198</td>
      <td>0.877091</td>
      <td>0.876956</td>
      <td>0.630653</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data=test_performances.melt(), x='variable', y='value')
plt.ylim(0.6,0.92)
```




    (0.6, 0.92)




    
![png](plot_validation_files/plot_validation_4_1.png)
    



```python
def make_features_numerical(df: pd.DataFrame) -> pd.DataFrame:
    '''Cannot train on boolean values.'''
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    return df
```


```python
df = pd.read_pickle('mouse_features_paper.pickle')
feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]
```


```python
def add_test_probabilities(df, cv_run: str, results_dir: str, folds = [0,1,2,3,4], model_type='sklearn_any'):

    for test in folds:
        # extract the features.
        df_X = df.loc[df[('Annotations', 'Fold')] == test, feature_columns]
        df_X = make_features_numerical(df_X)
        X =  df_X.values

        all_probs = []
        # predict from all the test models and average probabilities.
        for val in folds:
            if val == test:
                continue

            model = pickle.load(open(os.path.join(results_dir, cv_run, f'model_t{test}_v{val}.pkl'), 'rb'))
            
            # TODO adjust for SVM and bayes logreg
            if model_type == 'svc':
                probs = model.decision_function(X)
            else:
                probs = model.predict_proba(X)[:, 1]
            
            all_probs.append(probs)

        probs = np.stack(all_probs).mean(axis=0)
        df.loc[df[('Annotations', 'Fold')] == test, ('Predictions', cv_run)] = probs
```


```python
feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]
add_test_probabilities(df, 'cv_f_logreg', results_dir )
add_test_probabilities(df, 'cv_f_logreg_smote', results_dir )
add_test_probabilities(df, 'cv_rf', results_dir )
add_test_probabilities(df, 'cv_svc', results_dir, model_type='svc' )
feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('Chemical')]
add_test_probabilities(df, 'cv_chemical', results_dir )
```


```python
import seaborn as sns; sns.set()

def make_plottable(prediction, targets, source, model_name, auc=True):
    plot_data = pd.DataFrame({"prediction": prediction, "Target": targets.astype(int)})
    plot_data = plot_data.sort_values(by="prediction")[::-1]
    plot_data["Top X"] = np.arange(targets.shape[0])
    plot_data["Found"] = plot_data["Target"].cumsum()
    plot_data["source"] = source
    if auc:
        auc = sklearn.metrics.roc_auc_score(targets.astype(int),  prediction)
        model_name = "{} (AUC={:.3f})".format(model_name, auc)
    plot_data["model"] = model_name
    return plot_data

df_plot = pd.concat((
    make_plottable(df[('Predictions', 'cv_f_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Logistic regression"),
    make_plottable(df[('Predictions', 'cv_rf')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Random forest"),
    make_plottable(df[('Annotations', 'Intensity')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Null"),
    make_plottable(df[('Predictions', 'cv_svc')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Support vector machine"),
    make_plottable(df[('Predictions', 'cv_f_logreg_smote')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Logistic regression w/ SMOTE"),
    make_plottable(df[('Predictions', 'cv_chemical')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Chemical model"),

)).reset_index(drop=True) # need to reset the index for seaborn to work correctly.
```


```python
fig = plt.figure(figsize=(2 * 7, 6))
ax1, ax2 = fig.subplots(1, 2, sharey=True)
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot[df_plot["Top X"] < 1000], ax=ax1)
# sns.lineplot(x="Top X", y="Found", style="model", data=df[df["Top X"] < 1000], ax=ax2)
g = sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot, ax=ax2)
#g.axes.scatter(n_lpv, n_lpv_true, s=50, c='b', marker='x')
#g.axes.annotate('LPV', (n_lpv, n_lpv_true), xycoords='data',
#                xytext=(n_lpv * 1.25, n_lpv_true * 0.75), textcoords='data',
#                arrowprops=dict(arrowstyle= '->', color='k', lw=3.5, ls='--'))
#g.figure.savefig("figures/report/top_panel.pdf")
```


    
![png](plot_validation_files/plot_validation_10_0.png)
    

