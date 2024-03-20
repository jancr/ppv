```python
# core imports
import pickle
import os
import warnings
import pathlib

# 3rd party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import spearmanr
import logomaker 


sns.set(style='white')
```


```python
%matplotlib inline
```

```python
cd ../../ppv-data
```

## Functions used troughout the notebook

```python
def add_test_probabilities(df, cv_run: str, results_dir: str, folds = [0,1,2,3,4], model_type='sklearn_any', feature_columns=None):
    if feature_columns is None:
        # 1) Use only MS derived Features
        # 2) remove count based and leaking features.
        exclude_features = [
            (    'MS Count',             'start'),
            (    'MS Count',             'stop'),
            (    'MS Frequency',        'protein_coverage'),
            (    'MS Frequency',        'cluster_coverage'),
        ]
        feature_columns = df.columns[ (df.columns.get_level_values(0).str.startswith('MS')) & ~(df.columns.isin(exclude_features))]


    for test in folds:
        # extract the features.
        df_X_orig = df.loc[df[('Annotations', 'Fold')] == test, feature_columns]
        df_X = make_features_numerical(df_X_orig)
        X =  df_X.values

        all_probs = []
        # predict from all the test models and average probabilities.
        for val in folds:
            if val == test:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = pickle.load(open(os.path.join(results_dir, cv_run, f'model_t{test}_v{val}.pkl'), 'rb'))
            
            # TODO adjust for SVM and bayes logreg
            if model_type == 'svc':
                probs = model.decision_function(X)
            elif model_type == 'bayes':
                probs = model.predict(df_X_orig)
            else:
                probs = model.predict_proba(X)[:, 1]
            
            all_probs.append(probs)

        probs = np.stack(all_probs).mean(axis=0)
        df.loc[df[('Annotations', 'Fold')] == test, ('Predictions', cv_run)] = probs

        
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
 

```

```python
# df_assembly = pd.read_pickle('features/mouse_features_paper_assembly_sklearn_with_peptideranker.pickle')
```

```python
df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
```

## Figure 3b (degradation)

- Suggestion from reviewer: plot number of peptides per protein as a function of protein length
- Plot the number of fragments (shorter) per known peptide as a function of peptide length
- For our data: plot percentage of abundance of all fragments (length x or shorter) originating from the REAL peptide (the longest one) as a function of peptide length (try also with protein length). 



```python
df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
df_ileum = df.loc['Mouse Ileum']
df_ileum['Annotations', 'Length'] = df_ileum['Annotations', 'Sequence'].str.len()
```

```python
df_ileum.reset_index()['protein_id'].drop_duplicates().to_csv('proteins.txt.tmp', index=False)
```


```python
# df_list
```

```python
df_list = []
temp_df = df_ileum.loc[df_ileum['Annotations', 'Known']]['Annotations'][['Length', 'Fragment count']]
temp_df['Study'] = 'Ileum (this study)'
df_list.append(temp_df)
temp_df = df_larraufie_gastrointestinal.loc[df_larraufie_gastrointestinal['Known']][['Length', 'Fragment count']].copy()
temp_df['Study'] = 'Gastrointestinal tract (Larraufie 2019)'
df_list.append(temp_df)
temp_df = df_larraufie_enteroendocrine.loc[df_larraufie_enteroendocrine['Known']][['Length', 'Fragment count']].copy()
temp_df['Study'] = 'Enteroendocrine cells (Larraufie 2019)'
df_list.append(temp_df)
temp_df = df_galvin_enteroendocrine.loc[df_galvin_enteroendocrine['Known']][['Length', 'Fragment count']].copy()
temp_df['Study'] = 'Enteroendocrine cells (Galvin 2021)'
df_list.append(temp_df)

with pd.ExcelWriter('figures/paper/raw_data_figure_3b.xlsx') as writer:
    pd.concat(df_list).to_excel(writer, sheet_name='Figure 3b')


```

```python
# # Ulriks alternative
# # •	Plot the number of fragments (shorter) per known peptide as a function of peptide length
# # •	For our data: plot percentage of abundance of all fragments (length x or shorter) originating from the REAL peptide (the longest one) as a function of peptide length (try also with protein length). 
 
# for idx, row in df_ileum.loc[df_ileum['Annotations', 'Known']].iterrows():
#     protein_id, start_pre, stop_pre = idx

#     fragment_count = 0
#     for sub_idx, sub_row in df_ileum.loc[protein_id].iterrows():
#         start, stop = sub_idx

#         # require: complete overlap.
#         if start>= start_pre and stop <= stop_pre:
#             fragment_count +=1

#     df_ileum.loc[idx, ('Annotations', 'Fragment count')] = fragment_count
        
```

```python
# sns.scatterplot(x = df_ileum.loc[df_ileum['Annotations', 'Known']]['Annotations', 'Length'], y = df_ileum.loc[df_ileum['Annotations', 'Known']]['Annotations', 'Fragment count'])
```
```python
df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
df = df.loc[df['Annotations', 'Known']]
known_peptides = set(df['Annotations','Sequence'])
```


```python
larraufie_gastrointestinal = 'datasets/larraufie_gastrointestinal.tsv'
larraufie_enteroendocrine = 'datasets/larraufie_enteroendocrine.tsv'
galvin_enteroendocrine = 'datasets/galvin_enteroendocrine.tsv'


df = pd.read_csv(larraufie_gastrointestinal, sep='\t')
df = df.set_index(['accession', 'start', 'end'])
df['Length'] = df['pepseq'].str.len()

df['Known'] = False
for idx, row in df.iterrows():
    if row['pepseq'] in known_peptides:
        df.loc[idx, 'Known'] = True



for idx, row in df.loc[df['Known']].iterrows():
    protein_id, start_pre, stop_pre = idx

    fragment_count = 0
    for sub_idx, sub_row in df.loc[protein_id].iterrows():
        start, stop = sub_idx

        # require: complete overlap.
        if start>= start_pre and stop <= stop_pre:
            fragment_count +=1

    df.loc[idx, 'Fragment count'] = fragment_count


df_larraufie_gastrointestinal = df


df = pd.read_csv(larraufie_enteroendocrine, sep='\t')
df = df.set_index(['accession', 'start', 'end'])
df['Length'] = df['pepseq'].str.len()

df['Known'] = False
for idx, row in df.iterrows():
    if row['pepseq'] in known_peptides:
        df.loc[idx, 'Known'] = True



for idx, row in df.loc[df['Known']].iterrows():
    protein_id, start_pre, stop_pre = idx

    fragment_count = 0
    for sub_idx, sub_row in df.loc[protein_id].iterrows():
        start, stop = sub_idx

        # require: complete overlap.
        if start>= start_pre and stop <= stop_pre:
            fragment_count +=1

    df.loc[idx, 'Fragment count'] = fragment_count


df_larraufie_enteroendocrine = df
```

```python

```

```python
df = pd.read_csv(galvin_enteroendocrine, sep='\t')
df = df.set_index(['Protein Accession', 'Start', 'End'])
pepseq = df['Peptide sequence'].str.replace(r"\([^A-Z]*\)","", regex=True) # removes (xx.xx) mass annotations
df['pepseq'] = pepseq.str.slice(2,-2)# remove leading and trailing X. .X
df['Length'] = df['pepseq'].str.len()

df['Known'] = False
for idx, row in df.iterrows():
    if row['pepseq'] in known_peptides:
        df.loc[idx, 'Known'] = True



for idx, row in df.loc[df['Known']].iterrows():
    protein_id, start_pre, stop_pre = idx

    fragment_count = 0
    for sub_idx, sub_row in df.loc[protein_id].iterrows():
        start, stop = sub_idx

        # require: complete overlap.
        if start>= start_pre and stop <= stop_pre:
            fragment_count +=1

    df.loc[idx, 'Fragment count'] = fragment_count

    df_galvin_enteroendocrine = df


```

```python
df_ileum['Annotations', 'Fragment count log'] = np.log10(df_ileum['Annotations', 'Fragment count'])
df_larraufie_gastrointestinal['Fragment count log'] = np.log10(df_larraufie_gastrointestinal['Fragment count'])
df_larraufie_enteroendocrine['Fragment count log'] = np.log10(df_larraufie_enteroendocrine['Fragment count'])
df_galvin_enteroendocrine['Fragment count log'] = np.log10(df_galvin_enteroendocrine['Fragment count'])
```


```python
fig, ax = plt.subplots(1,4, figsize=(16,4), sharey=True, sharex=True)

sns.scatterplot(data=df_larraufie_gastrointestinal.loc[df_larraufie_gastrointestinal['Known']], x='Length', y='Fragment count log', ax= ax[0])
sns.scatterplot(data=df_larraufie_enteroendocrine.loc[df_larraufie_enteroendocrine['Known']], x='Length', y='Fragment count log', ax= ax[1])
sns.scatterplot(data=df_galvin_enteroendocrine.loc[df_galvin_enteroendocrine['Known']], x='Length', y='Fragment count log', ax= ax[2])
sns.scatterplot(x = df_ileum.loc[df_ileum['Annotations', 'Known']]['Annotations', 'Length'], y = df_ileum.loc[df_ileum['Annotations', 'Known']]['Annotations', 'Fragment count log'], ax=ax[3])


ax[0].set_title('Gastrointestinal tract (Larraufie 2019)')
ax[1].set_title('Enteroendocrine cells (Larraufie 2019)')
ax[2].set_title('Enteroendocrine cells (Galvin 2021)')
ax[3].set_title('Ileum (this study)')

for a in ax:
    a.set_ylabel('Fragment count (log10)')
    a.set_xlabel('Peptide length')

plt.tight_layout()

plt.savefig(f"figures/paper/figure_3b.svg")
plt.savefig(f"figures/paper/figure_3b.pdf")
```
## Figure 4c


```python
sns.set(style='white')
```


```python
def make_features_numerical(df: pd.DataFrame) -> pd.DataFrame:
    '''Cannot train on boolean values.'''
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)
    return df

df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
# feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]



results_dir = 'nested_cv'
# exclude_features = [
#     (    'MS Count',             'start'),
#     (    'MS Count',             'stop'),
#     (    'MS Frequency',        'protein_coverage'),
#     (    'MS Frequency',        'cluster_coverage'),
# ]
# feature_columns = df.columns[ (df.columns.get_level_values(0).str.startswith('MS')) & ~(df.columns.isin(exclude_features))]
add_test_probabilities(df, 'cv_f_logreg', results_dir )
# add_test_probabilities(df, 'cv_max_42aa', results_dir )
# add_test_probabilities(df, 'cv_max_pos_30aa', results_dir )
#add_test_probabilities(df, 'cv_f_logreg_smote', results_dir )
# add_test_probabilities(df, 'cv_rf', results_dir )
#df[('Predictions', 'rf_old')] = df[('Predictions', 'cv_rf')]
# add_test_probabilities(df, 'cv_svc', results_dir, model_type='svc' )
#add_test_probabilities(df, 'cv_bayes_logreg', results_dir, model_type='bayes')
# feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('Chemical')]
# add_test_probabilities(df, 'cv_chemical', results_dir, feature_colums=feature_columns)

#results_dir = 'bayes_opt'
#feature_columns = df.columns[df.columns.get_level_values(0).str.startswith('MS')]
#add_test_probabilities(df, 'cv_elasticnet', results_dir )
#add_test_probabilities(df, 'cv_rf', results_dir )


df_plot = pd.concat((
    make_plottable(df[('Predictions', 'cv_f_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "PPV"),
    #make_plottable(df[('Predictions', 'cv_bayes_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Bayesian logistic regression"),
    # make_plottable(df[('Predictions', 'cv_rf')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Random forest"),
    make_plottable(df[('Annotations', 'Intensity')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Null"),
    # make_plottable(df[('Predictions', 'cv_svc')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Support vector machine"),
    #make_plottable(df[('Predictions', 'cv_f_logreg_smote')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Logistic regression w/ SMOTE"),
    # make_plottable(df[('Predictions', 'cv_chemical')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Chemical model"),
)).reset_index(drop=True) # need to reset the index for seaborn to work correctly.


fig = plt.figure(figsize=(2 * 7, 6))
ax1, ax2 = fig.subplots(1, 2, sharey=True)
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot[df_plot["Top X"] < 300], ax=ax1)
# sns.lineplot(x="Top X", y="Found", style="model", data=df[df["Top X"] < 1000], ax=ax2)
g = sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot, ax=ax2)
#g.axes.scatter(n_lpv, n_lpv_true, s=50, c='b', marker='x')
#g.axes.annotate('LPV', (n_lpv, n_lpv_true), xycoords='data',
#                xytext=(n_lpv * 1.25, n_lpv_true * 0.75), textcoords='data',
#                arrowprops=dict(arrowstyle= '->', color='k', lw=3.5, ls='--'))
#g.figure.savefig("figures/paper/figure_4c_complete_curve.png")
#g.figure.savefig("figures/paper/figure_4c_complete_curve.svg")
```

```python

```

```python
# df['Annotations', 'Length'] = pd.cut(df['Annotations']['Sequence'].str.len(), [0,5,10,15,20,25,30,35,40,45,50,55,60,65], labels=[5,10,15,20,25,30,35,40,45,50,55,60,65])
```


```python
# sns.boxplot(x=df['Annotations']['Length'], y= df['Predictions']['cv_max_pos_30aa'])
```
```python
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# sns.boxplot(data = df.loc[~df['Annotations', 'Known'].astype(bool)], x=('Annotations', 'Length'), y=('Predictions', 'cv_max_pos_30aa'))
# plt.subplot(1,2,2)
# sns.boxplot(data = df.loc[~df['Annotations', 'Known'].astype(bool)], x=('Annotations', 'Length'), y=('Predictions', 'cv_f_logreg'))
```

```python
# # sns.scatterplot(data = df.loc[df['Annotations', 'Known'].astype(bool) & (df['Annotations','Sequence'].str.len()>30)], x=('Predictions', 'cv_f_logreg'), y=('Predictions', 'cv_max_pos_30aa'))

# from scipy.special import logit

# x = logit(df.loc[df['Annotations', 'Known'].astype(bool) & (df['Annotations','Sequence'].str.len()>30)]['Predictions', 'cv_f_logreg'])
# y = logit(df.loc[df['Annotations', 'Known'].astype(bool) & (df['Annotations','Sequence'].str.len()>30)]['Predictions', 'cv_max_pos_30aa'])

# sns.scatterplot(x=x, y=y)
# spearmanr(x,y)
```
```python

```


```python
# df['Annotations', 'difference'] = df['Predictions', 'cv_f_logreg'] -df['Predictions', 'cv_max_42aa']
```





```python
n_lpv = df['Annotations','LPV'].sum()
n_lpv_true = (df['Annotations','LPV'].astype(bool) & df['Annotations', 'Known'].astype(bool)).sum()
```


```python
df_plot = pd.concat((
    make_plottable(df[('Predictions', 'cv_f_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "PPV (logistic regression)"),
    # make_plottable(df[('Predictions', 'cv_bayes_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Bayesian logistic regression"),
    # make_plottable(df[('Predictions', 'cv_rf')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Random forest"),
    # make_plottable(df[('Predictions', 'cv_svc')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Support vector machine"),
    # make_plottable(df[('Predictions', 'cv_f_logreg_smote')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Logistic regression w/ SMOTE"),

    # make_plottable(df[('Predictions', 'cv_elasticnet')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Elastic net"),
    make_plottable(df[('Annotations', 'Intensity')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Null"),
    #make_plottable(df[('Predictions', 'cv_rf')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Random forest bayes opt"),
    make_plottable(df[('Predictions', 'PeptideRanker')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "PeptideRanker"),
)).reset_index(drop=True) # need to reset the index for seaborn to work correctly.

fig = plt.figure(figsize=(2 * 7, 6))
ax1, ax2 = fig.subplots(1, 2, sharey=False)
sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot[df_plot["Top X"] < 300], ax=ax1, legend=False)
g = sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot, ax=ax2)

g.axes.scatter(n_lpv, n_lpv_true, s=50, c='red', marker='o')
g.axes.annotate('LPV', (n_lpv, n_lpv_true), xycoords='data',
                xytext=(n_lpv * 1.05, n_lpv_true * 0.95), textcoords='data',
                )#arrowprops=dict(arrowstyle= '->', color='k', lw=3.5, ls='--'))

g.figure.savefig("figures/supplement/roc_curves_ml_models.png")
g.figure.savefig("figures/supplement/roc_curves_ml_models.svg")
```
```python
df_plot = pd.concat((
    make_plottable(df[('Predictions', 'cv_f_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "PPV (logistic regression)"),
    #make_plottable(df[('Predictions', 'cv_bayes_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Bayesian logistic regression"),
    #make_plottable(df[('Predictions', 'cv_rf')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Random forest"),
    #make_plottable(df[('Predictions', 'cv_svc')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Support vector machine"),
    #make_plottable(df[('Predictions', 'cv_f_logreg_smote')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Logistic regression w/ SMOTE"),
    #make_plottable(df[('Predictions', 'cv_chemical')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Chemical model"),

    #make_plottable(df[('Predictions', 'cv_elasticnet')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Elastic net"),
    #make_plottable(df[('Annotations', 'Intensity')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Null"),
    #make_plottable(df[('Predictions', 'cv_rf')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "Random forest bayes opt"),
    make_plottable(df[('Predictions', 'PeptideRanker')].values, df[('Annotations', 'Known')].cat.codes.values, "Observed", "PeptideRanker"),
)).reset_index(drop=True) # need to reset the index for seaborn to work correctly.

fig = plt.figure(figsize=(2 * 7, 6))
ax1, ax2 = fig.subplots(1, 2, sharey=False)
sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot[df_plot["Top X"] < 300], ax=ax1, legend=False)
g = sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot, ax=ax2)

g.figure.savefig("figures/supplement/roc_curves_ppv_peptideranker.png")
g.figure.savefig("figures/supplement/roc_curves_ppv_peptideranker.svg")
```
```python

```


```python
df = pd.read_pickle('features/mouse_features_paper_assembly_sklearn_with_peptideranker.pickle')
results_dir = 'nested_cv_assembly'
add_test_probabilities(df, 'cv_f_logreg', results_dir )
df_plot_assembly = make_plottable(df[('Predictions', 'cv_f_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, 
                                  "Observed", "PPV assembly", auc=False)

df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
results_dir = 'nested_cv'
add_test_probabilities(df, 'cv_f_logreg', results_dir )
df_plot = make_plottable(df[('Predictions', 'cv_f_logreg')].values, df[('Annotations', 'Known')].cat.codes.values, 
                         "Observed", "PPV", auc=False)

df_plot_assembly['Top X %'] = df_plot_assembly['Top X'] / len(df_plot_assembly) * 100
df_plot['Top X %'] = df_plot['Top X'] / len(df_plot) * 100
df_plot = pd.concat([df_plot, df_plot_assembly]).reset_index(drop=True)

fig = plt.figure(figsize=(2 * 7, 6))
ax1, ax2 = fig.subplots(1, 2, sharey=False)
sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot[df_plot["Top X"] < 300], ax=ax1, legend=False)
g = sns.lineplot(x="Top X", y="Found", hue="model", data=df_plot, ax=ax2)

g.figure.savefig("figures/supplement/roc_curves_assembly.png")
g.figure.savefig("figures/supplement/roc_curves_assembly.svg")
```
## Figure 4

```python
exclude_features = [
    (    'MS Count',             'start'),
    (    'MS Count',             'stop'),
    (    'MS Frequency',        'protein_coverage'),
    (    'MS Frequency',        'cluster_coverage'),
]
feature_columns = df.columns[ (df.columns.get_level_values(0).str.startswith('MS')) & ~(df.columns.isin(exclude_features))]
models = []
model_names = []
for t in range(5):
    for v in range(5):
        if not t ==v:
            models.append(pickle.load(open(os.path.join('nested_cv', 'cv_f_logreg', f'model_t{t}_v{v}.pkl'), 'rb')))
            model_names.append((t,v))


coef = [model.named_steps['logreg'].coef_ for model in models]
coef_df = pd.DataFrame(np.concatenate(coef), columns=feature_columns)

coef_raw = [model.named_steps['logreg'].coef_ / np.sqrt(model.named_steps['scaler'].var_) for model in models]
coef_raw_df = pd.DataFrame(np.concatenate(coef_raw), columns=feature_columns)

coef_df.columns = [' '.join(col).strip() for col in coef_df.columns.values]
coef_raw_df.columns= [' '.join(col).strip() for col in coef_raw_df.columns.values]
```


## Figure 4b


```python
order = coef_df.mean(axis=0).sort_values().index.tolist()
```


```python
sns.set_theme(style='white')
```


```python
fig, axs = plt.subplots(1,2,figsize=(8*2,5))

axs[0].axvline(0, c='grey', linestyle='--', zorder=0)
axs[1].axvline(0, c='grey', linestyle='--', zorder=0)
sns.boxplot(data=coef_raw_df.melt(), x='value', y='variable',color='blue', ax=axs[0], order = order, whis=5)
#sns.stripplot(data=coef_raw_df.melt(), x='value', y='variable',color='grey', ax=axs[0], order = order)

sns.boxplot(data=coef_df.melt(), x='value', y='variable',color='blue', ax=axs[1], order = order, whis=5)
axs[0].set_xlim(-4.5,4.5)
axs[1].set_xlim(-2,2)

fig.tight_layout()
plt.savefig(f"figures/paper/figure_4b.svg")
plt.savefig(f"figures/paper/figure_4b.pdf")
```
```python
with pd.ExcelWriter('figures/paper/raw_data_figure_4b.xlsx') as writer:
    coef_df.to_excel(writer, sheet_name='Figure 4b')
```

```python
ppv_data["Predictions"]
```

## Supplement Figure 5.1

```python
order = coef_df.mean(axis=0).sort_values().index.tolist()
exclude_features = [
    (    'MS Count',             'start'),
    (    'MS Count',             'stop'),
    (    'MS Frequency',        'protein_coverage'),
    (    'MS Frequency',        'cluster_coverage'),
]
feature_columns = ppv_data.columns[(ppv_data.columns.get_level_values(0).str.startswith('MS')) & ~(ppv_data.columns.isin(exclude_features))]
models = []
model_names = []
for t in range(5):
    for v in range(5):
        if not t ==v:
            models.append(pickle.load(open(os.path.join('nested_cv_assembly', 'cv_f_logreg', f'model_t{t}_v{v}.pkl'), 'rb')))
            model_names.append((t,v))


coef = [model.named_steps['logreg'].coef_ for model in models]
coef_df = pd.DataFrame(np.concatenate(coef), columns=feature_columns)

coef_raw = [model.named_steps['logreg'].coef_ / np.sqrt(model.named_steps['scaler'].var_) for model in models]
coef_raw_df = pd.DataFrame(np.concatenate(coef_raw), columns=feature_columns)

coef_df.columns = [' '.join(col).strip() for col in coef_df.columns.values]
coef_raw_df.columns= [' '.join(col).strip() for col in coef_raw_df.columns.values]
```

```python
fig, axs = plt.subplots(1,2,figsize=(8*2,5))

axs[0].axvline(0, c='grey', linestyle='--', zorder=0)
axs[1].axvline(0, c='grey', linestyle='--', zorder=0)
sns.boxplot(data=coef_raw_df.melt(), x='value', y='variable',color='blue', ax=axs[0], order = order, whis=5)
#sns.stripplot(data=coef_raw_df.melt(), x='value', y='variable',color='grey', ax=axs[0], order = order)

sns.boxplot(data=coef_df.melt(), x='value', y='variable',color='blue', ax=axs[1], order = order, whis=5)
axs[0].set_xlim(-4.5,4.5)
axs[1].set_xlim(-2,2)

fig.tight_layout()
plt.savefig(f"figures/supplement/assembly_features.svg")
plt.savefig(f"figures/supplement/assembly_features.pdf")
```

## Figure 5a


```python
obs_data = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
ppv_data = pd.read_pickle('features/mouse_features_paper_assembly_sklearn_with_peptideranker.pickle')
results_dir = 'nested_cv'
add_test_probabilities(obs_data, 'cv_f_logreg', results_dir )

```


```python

```

```python
import logomaker

def create_pfm(df, *, by='Counts'):
    annotations = df["Annotations"].copy()
    if by == "Counts":
        annotations[by] = 1
    weights = annotations[by]
    
    aa = list('ACDEFGHIKLMNPQRSTVWY')
    n_counts = pd.DataFrame(np.zeros((4, len(aa))), columns=aa, index=[-4, -3, -2, -1])
    c_counts = pd.DataFrame(np.zeros((4, len(aa))), columns=aa, index=[1, 2, 3, 4])
    _iter = ((n_counts, annotations[by].values, annotations["N Flanking"].values, [-4, -3, -2, -1]),
             (c_counts, annotations[by].values, annotations["C Flanking"].values, [1, 2, 3, 4]))
    for counts, weights, flanking_region, indexer in _iter:
        for w, seq in zip(weights, flanking_region):
            for index, aa in zip(indexer, seq):
                if aa in '_U':
                    continue
                counts.loc[index, aa] += w
    return n_counts, c_counts


def norm(counts, pseudo_count=0.1):
    counts = counts + pseudo_count
    return (counts.T / counts.sum(axis=1)).T


def kl(p, q):
    return (p * np.log2(p / q)).sum(axis=1)


def calc_height(p_count, q_count):
    p = norm(p_count)
    q = norm(q_count)
    I = kl(p, q)
    return (p.T * I).T
    
    
def create_logo_on_axis(height, ax, terminal, max_height, color_scheme=None):
    ax.set_title(f"{terminal}-Terminal Flanking Region", fontsize=14)
    logomaker.Logo(height, font_name='Arial Rounded MT Bold', ax=ax, color_scheme=color_scheme)
    ax.xaxis.grid(False)
    ax.patch.set_visible(False)
    ax.set_ylim(0, max_height)
    
    
def create_logo_plot(n_fg_counts, c_fg_counts, n_bg_counts, c_bg_counts, tissue="Mouse Brain", *, color_scheme=None, max_height=None):
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"{tissue} - KL Logo Plot", fontsize=16)
    axes = fig.subplots(1, 2)
    n_height = calc_height(n_fg_counts, n_bg_counts)
    c_height = calc_height(c_fg_counts, c_bg_counts)
    if max_height is None:
        max_height = max(n_height.sum(axis=1).max(), c_height.sum(axis=1).max())
    axes[1].get_yaxis().set_visible(False)
    create_logo_on_axis(n_height, axes[0], "N", max_height, color_scheme)
    create_logo_on_axis(c_height, axes[1], "C", max_height, color_scheme)
    return fig, max_height

def get_aa_counts(fasta_file, amino_acids):
    amino_acids = set(amino_acids)
    counter = {aa: 0 for aa in amino_acids}
    for line in open(fasta_file):
        if line.startswith('>'):
            continue
        for aa in line.strip():
            if aa in amino_acids:
                counter[aa] += 1
    return counter
    total = sum(counter.values())
    return {aa: count / total for (aa, count) in counter.items()}



amino_acids = 'ARNDBCEQZGHILKMFPSTWYV'
color_scheme = {aa: '#ececec' for aa in amino_acids}
color_scheme.update({
    "R": "#9ca2d1",
    "K": "#9ca2d1",
    "G": "#92c4e0"
})
df = obs_data.copy()
df["Annotations", "Prediction"] =  obs_data['Predictions', 'cv_f_logreg']

n_total = 500
for tissue in {i[0] for i in obs_data.index}:
    # PPV
    df_tissue = df.loc[tissue].sort_values(("Annotations", "Prediction"), ascending=False)
    known_tissue = df_tissue["Annotations", "Known"].astype(bool)
    heights = []
    for _df in (df_tissue, df_tissue[~known_tissue]):
        # PPV
        top = _df.head(n_total)
        rest = _df.tail(-n_total)
        n_fg, c_fg = create_pfm(top)
        n_bg, c_bg = create_pfm(rest)
        fig, height = create_logo_plot(n_fg, c_fg, n_bg, c_bg, tissue, color_scheme=color_scheme)
        heights.append(height)
        extra = ''
        if _df is not df_tissue:
            extra = "_without_known"
        fig.savefig(f"figures/paper/figure_3h_and_supplement_ppv_{tissue}{extra}.svg")
        fig.savefig(f"figures/paper/figure_3h_and_supplement_ppv_{tissue}{extra}.pdf")
        
    # LPV
    df_tissue = ppv_data.loc[tissue]
    known_tissue = df_tissue["Annotations", "Known"].astype(bool)
    lpv_selector = df_tissue["Annotations", "LPV"]
    obs_selector = df_tissue["MS Bool", "observed"]
    for max_height, _df in zip(heights, (df_tissue, df_tissue[~known_tissue])):
        # PPV
        lpv = df_tissue[lpv_selector]
        not_lpv = df_tissue[~lpv_selector & obs_selector]
        n_lpv, c_lpv = create_pfm(lpv)
        n_not_lpv, c_not_lpv = create_pfm(not_lpv)
        
        fig, _ = create_logo_plot(n_lpv, c_lpv, n_not_lpv, c_not_lpv, tissue, 
                                  color_scheme=color_scheme, max_height=max_height)
        extra = ''
        if _df is not df_tissue:
            extra = "_without_known"
        fig.savefig(f"figures/paper/figure_3h_and_supplement_lpv_{tissue}{extra}.svg")
        fig.savefig(f"figures/paper/figure_3h_and_supplement_lpv_{tissue}{extra}.pdf")

```

```python
"https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Clength%2Cgene_names%2Corganism_name&format=tsv&query=%28cc_scl_term%3ASL-0243%29%20AND%20%28organism_id%3A10090%29"

```

```python
uniprot_secreted = pd.read_csv("https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Clength%2Cgene_names%2Corganism_name&format=tsv&query=%28cc_scl_term%3ASL-0243%29%20AND%20%28organism_id%3A10090%29", sep='\t')

```

```python
# uniprot_secreted = pd.read_csv('https://www.uniprot.org/uniprot/?query=locations:(location:%22Secreted%20[SL-0243]%22)&fil=organism%3A%22Mus+musculus+(Mouse)+[10090]%22&format=tab#', sep='\t')
uniprot_secreted = pd.read_csv("https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Clength%2Cgene_names%2Corganism_name&format=tsv&query=%28cc_scl_term%3ASL-0243%29%20AND%20%28organism_id%3A10090%29", sep='\t')

is_secreted = set(uniprot_secreted["Entry"])
def create_secreted(index):
    secreted = pd.Series(False, index=index, dtype=bool)
    for campaign_id, protein_id, start ,stop in index:
    #for campaign_id, pep_id in index.peptidomics.iter_index():
        _index = campaign_id, protein_id, start, stop
        secreted[_index] = protein_id in is_secreted
    return secreted
    
obs_data["Annotations", "Secreted"] = create_secreted(obs_data.index)
```


```python
uniprot_proteome = pd.read_csv("https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Clength%2Cgene_names%2Corganism_name&format=tsv&query=%28xref%3Aproteomes-up000000589%29", sep='\t')
uniprot_proteome_secreted = pd.read_csv("https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Clength%2Cgene_names%2Corganism_name&format=tsv&query=%28cc_scl_term%3ASL-0243%29%20AND%20%28xref%3Aproteomes-up000000589%29", sep='\t')
```

```python
proteome_secreted = len(uniprot_proteome_secreted)/len(uniprot_proteome) * 100
peptidome_secreted = obs_data['Annotations', 'Secreted'].sum() / len(obs_data) * 100

sub_df = obs_data.loc[obs_data['Predictions', 'cv_f_logreg']>0.01]
high_secreted = sub_df['Annotations', 'Secreted'].sum() / len(sub_df) * 100


sub_df = obs_data.nlargest(200, ('Predictions', 'cv_f_logreg'))
top_secreted = sub_df['Annotations', 'Secreted'].sum() / len(sub_df) * 100
```


```python
pd.Series({'Prot.': proteome_secreted, 'Pept.': peptidome_secreted, 'PPV>\n0.01': high_secreted, 'PPV\nTop200':top_secreted}).plot(kind='bar', rot=0)

plt.savefig("figures/paper/5b.pdf")
plt.savefig("figures/paper/5b.svg")
```
```python
obs_data_no_known = obs_data.loc[~obs_data["Annotations", "Known"].astype(bool)]
proteome_secreted = len(uniprot_proteome_secreted)/len(uniprot_proteome) * 100
peptidome_secreted = obs_data_no_known['Annotations', 'Secreted'].sum() / len(obs_data) * 100

sub_df = obs_data_no_known.loc[obs_data_no_known['Predictions', 'cv_f_logreg']>0.01]
high_secreted = sub_df['Annotations', 'Secreted'].sum() / len(sub_df) * 100


sub_df = obs_data_no_known.nlargest(200, ('Predictions', 'cv_f_logreg'))
top_secreted = sub_df['Annotations', 'Secreted'].sum() / len(sub_df) * 100
                                 
pd.Series({'Prot.': proteome_secreted, 'Pept.': peptidome_secreted, 'PPV>\n0.01': high_secreted, 'PPV\nTop200':top_secreted}).plot(kind='bar', rot=0)

```

```python
peptidome_secreted
```

```python
sub_df = obs_data.nlargest(200, ('Predictions', 'cv_f_logreg'))
sub_df['Annotations', 'Secreted'].sum() / len(sub_df) * 100
top_secreted

```

```python
sub_df = obs_data.loc[~obs_data["Annotations", "Known"].astype(bool)].nlargest(200, ('Predictions', 'cv_f_logreg'))
sub_df['Annotations', 'Secreted'].sum() / len(sub_df) * 100

```

## Figure 5c


```python
import matplotlib as mpl
mpl.style.use('default')
```


```python
fig = plt.figure(figsize=(8,5))
ax = plt.gca()
sns.kdeplot(np.log10(obs_data['Predictions', 'cv_f_logreg']), ax=ax)

rec = plt.Rectangle((-2,0),2,0.05, alpha=0.4, facecolor='grey', linewidth=1, edgecolor='black')
ax.add_patch(rec)

below_001 = (obs_data['Predictions', 'cv_f_logreg']<=0.01).sum() /len(obs_data)
ax.text(-4.4, 0.15, f'{below_001:.2%}')
sns.despine()

kde_x = ax.lines[0].get_xdata()
kde_y = ax.lines[0].get_ydata()
plt.fill_between(kde_x[kde_x<=-1.98], kde_y[kde_x<=-1.98], alpha=0.5, label='>0.01')
plt.fill_between(kde_x[kde_x>=-2], kde_y[kde_x>=-2], color='yellow', alpha=0.5, label='>0.01')

plt.savefig("figures/paper/5c.pdf")
plt.savefig("figures/paper/5c.svg")

```
```python
print(f"Above 0.01: {(obs_data['Predictions', 'cv_f_logreg']>0.01).sum() /len(obs_data) *100}")
print(f"Above 0.05: {(obs_data['Predictions', 'cv_f_logreg']>0.05).sum() /len(obs_data) *100}")
```

```python
kde_x = ax.lines[0].get_xdata()
kde_y = ax.lines[0].get_ydata()
```


```python
plt.plot(kde_x[kde_x>0.01], kde_y[kde_x>0.01])

thr = 0.05
plt.fill_between(kde_x[kde_x>0.01], kde_y[kde_x>0.01], color='yellow', alpha=0.5, label='>0.01')
plt.fill_between(kde_x[kde_x>thr], kde_y[kde_x>thr], color='orange', alpha=0.5, label='>0.05')
ax = plt.gca()
ax.set_xscale('log')
ax.xaxis.get_ticklocs(minor=True)
ax.minorticks_on()
ax.yaxis.set_tick_params(which='minor', bottom=False)
plt.xlim(0.01, 1)
plt.ylim(0,)
sns.despine()
plt.legend(title='PPV score')


perc_001 = f"{((obs_data['Predictions', 'cv_f_logreg']>0.01).sum() - (obs_data['Predictions', 'cv_f_logreg']>0.05).sum()) /len(obs_data):.2%}"
perc_005 = f"{(obs_data['Predictions', 'cv_f_logreg']>0.05).sum() /len(obs_data):.2%}"
ax.text(0.03, 0.02, perc_001)
ax.plot((0.02, 0.03), (0.015, 0.019), 'black', linestyle='-')

ax.text(0.1, 0.005, perc_005)
ax.plot((0.07, 0.1), (0.001, 0.004), 'black', linestyle='-')


plt.savefig("figures/paper/5c_zoom.pdf")
plt.savefig("figures/paper/5c_zoom.svg")
```
## Figure 7a


```python
results_dir = 'nested_cv'
obs_data = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
feature_columns = obs_data.columns[obs_data.columns.get_level_values(0).str.startswith('MS') & ~(obs_data.columns.isin(exclude_features))]
add_test_probabilities(obs_data, 'cv_f_logreg', results_dir )

obs_data['Annotations', 'C Inside'] = obs_data['Annotations', 'Sequence'].apply(lambda x: x[-4:])
```


```python
def create_pfm(df, *, by='Counts', n_name="N Flanking", c_name="C Flanking"):
    annotations = df["Annotations"].copy()
    if by == "Counts":
        annotations[by] = 1
    weights = annotations[by]
    
    aa = list('ACDEFGHIKLMNPQRSTVWY')
    n_counts = pd.DataFrame(np.zeros((4, len(aa))), columns=aa, index=[-4, -3, -2, -1])
    c_counts = pd.DataFrame(np.zeros((4, len(aa))), columns=aa, index=[1, 2, 3, 4])
    _iter = ((n_counts, annotations[by].values, annotations[n_name].values, [-4, -3, -2, -1]),
             (c_counts, annotations[by].values, annotations[c_name].values, [1, 2, 3, 4]))
    for counts, weights, flanking_region, indexer in _iter:
        for w, seq in zip(weights, flanking_region):
            for index, aa in zip(indexer, seq):
                if aa in '_U':
                    continue
                counts.loc[index, aa] += w
    return n_counts, c_counts


def norm(counts, pseudo_count=0.1):
    counts = counts + pseudo_count
    return (counts.T / counts.sum(axis=1)).T
    
def kl(p, q):
    return (p * np.log2(p / q)).sum(axis=1)


def calc_kl_height(p_count, q_count):
    p = norm(p_count)
    q = norm(q_count)
    I = kl(p, q)
    return (p.T * I).T


def calc_log_ratio_height(fg_count, bg_count):
    return np.log2(norm(fg_count)) - np.log2(norm(bg_count))
    
    
def calc_diff_height(fg_count, bg_count):
    return norm(fg_count) - norm(bg_count)

def create_logo_on_axis(height, ax, title, ylim=None, color_scheme=None):
#     ax.set_title(f"{terminal}-Terminal Flanking Region", fontsize=14)
    ax.set_title(title, fontsize=14)
    logomaker.Logo(height, font_name='Arial Rounded MT Bold', ax=ax, color_scheme=color_scheme, flip_below=False)
    ax.xaxis.grid(False)
    ax.patch.set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)

   
def create_logo_plot(n_fg_counts, c_fg_counts, n_bg_counts, c_bg_counts, tissue="Mouse Brain", *,
                     color_scheme=None, ylim=None, n_title="N-Terminal Flanking Region",
                     c_title="C-Terminal Flanking Region", name="KL", calc_height=calc_kl_height):
                     
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"{tissue} - {name} Logo Plot", fontsize=16)
    axes = fig.subplots(1, 2)
    
    n_height = calc_height(n_fg_counts, n_bg_counts)
    c_height = calc_height(c_fg_counts, c_bg_counts)
    
    if ylim is None:
        ymax = max(c_height[(0 < c_height)].sum(axis=1).max(), n_height[(0 < n_height)].sum(axis=1).max())
        ymin = min(c_height[(c_height < 0)].sum(axis=1).min(), n_height[(n_height < 0)].sum(axis=1).min())
        ylim = (ymin, ymax)
        
    axes[1].get_yaxis().set_visible(False)
    create_logo_on_axis(n_height, axes[0], n_title, ylim, color_scheme)
    create_logo_on_axis(c_height, axes[1], c_title, ylim, color_scheme)
    return fig, ylim

def get_aa_counts(fasta_file, amino_acids):
    amino_acids = set(amino_acids)
    counter = {aa: 0 for aa in amino_acids}
    for line in open(fasta_file):
        if line.startswith('>'):
            continue
        for aa in line.strip():
            if aa in amino_acids:
                counter[aa] += 1
    return counter

amino_acids = 'ARNDBCEQZGHILKMFPSTWYV'
color_scheme = {aa: '#ececec' for aa in amino_acids}
color_scheme.update({
    "R": "#9ca2d1",
    "K": "#9ca2d1",
    "G": "#92c4e0"
})

def _create_figs(fg, bg, tissue, ylims=(None, None), *, n_name="N Flanking", c_name="C Flanking",
                color_scheme=color_scheme):
    n_fg, c_fg = create_pfm(fg, n_name=n_name, c_name=c_name)
    n_bg, c_bg = create_pfm(bg, n_name=n_name, c_name=c_name)
    names = ("KL", "Change in Percent")
    height_functions = [calc_kl_height, lambda p, q: 100 * calc_diff_height(p, q)]
    return list(zip(*[create_logo_plot(n_fg, c_fg, n_bg, c_bg, tissue.title(), calc_height=height_function,
                            name=name, color_scheme=color_scheme, ylim=ylim)
            for name, height_function, ylim in zip(names, height_functions, ylims)]))

def _save(figs, name, tissue, extra='', types=('kl', 'diff')):
    for fig, type_ in zip(figs, types):
        for ext in ('svg', 'pdf'):
            folder = pathlib.Path(f"../figures/logo/{type_}/{ext}")
            folder.mkdir(exist_ok=True, parents=True)
            fig.savefig(folder / f"{name}_{tissue.lower()}{extra}.{ext}")

```


```python
amidation_selector = obs_data["MS Frequency", "amidation"] > 0.75
g_selector = obs_data["Annotations", "C Flanking"].apply(lambda x: x[0] == 'G')
# obs_data.loc[~g_selector & amidation_selector]["Annotations"]


known_selector = obs_data["Annotations", "Known"].astype(bool)
n_total = 100
extra = '_without_known'

df_no_g = obs_data.loc[~g_selector & ~known_selector]
amidation_selector = df_no_g["MS Frequency", "amidation"] > 0.75

# amidated vs not amidated
alt_amidated = df_no_g.loc[amidation_selector]
not_amidated = df_no_g.loc[~amidation_selector]
figs1, _ = _create_figs(alt_amidated, not_amidated, 'global', n_name="C Inside", color_scheme=None)
_save(figs1, 'alt_am_vs_not_am', 'global', extra)

name = 'alt_am_vs_not_am'
for fig, type_ in zip(figs1, ('kl', 'diff')):
    for ext in ('svg', 'pdf'):
        folder = pathlib.Path("figures/paper")
        fig.savefig(folder / f"7a.{ext}")
```
## Figure 7b


```python
inside = 'EP' # + 'D'
outside = 'PE' # + 'D'

total_am = df_no_g[amidation_selector].shape[0]
total_not_am = df_no_g[~amidation_selector].shape[0]

        
with open('figures/paper/7b.txt', 'w') as f:
    f.write('Global\n')
    def _print(*args):
        print(*args)
        print(*args, file=f)

    for _in in inside:
        for _out in outside:
            e_inside_selector = df_no_g["Annotations", "C Inside"].apply(lambda x: x[3] == _in)
            p_outside_selector = df_no_g["Annotations", "C Flanking"].apply(lambda x: x[0] == _out)

            am_io = df_no_g.loc[amidation_selector & e_inside_selector & p_outside_selector].shape[0] / total_am
            am_i = df_no_g.loc[amidation_selector & e_inside_selector].shape[0] / total_am
            am_o = df_no_g.loc[amidation_selector & p_outside_selector].shape[0] / total_am

            not_am_io = df_no_g.loc[~amidation_selector & e_inside_selector & p_outside_selector].shape[0]  / total_not_am
            not_am_i = df_no_g.loc[~amidation_selector & e_inside_selector].shape[0] / total_not_am
            not_am_o = df_no_g.loc[~amidation_selector & p_outside_selector].shape[0]  / total_not_am

            _print(f'{_in}|*: am={100 * am_i:4.2f}% bg={100 * not_am_i:4.2f}%, enrichment={am_i / not_am_i:4.2f}')
            _print(f'*|{_out}: am={100 * am_o:4.2f}% bg={100 * not_am_o:4.2f}%, enrichment={am_o / not_am_o:4.2f}')
            _print(f'{_in}|{_out}: am={100 * am_io:4.2f}% bg={100 * not_am_io:4.2f}%, enrichment={am_io / not_am_io:4.2f}')
            both = am_io / not_am_io
            expected_both = am_i * am_o / (not_am_i * not_am_o)
            _print(f'---: expected={expected_both:4.2f} observed={both:4.2f}, extra_enrichment={both / expected_both:4.2f}')
            _print()

```

```python
# inside = 'EP' # + 'D'
# outside = 'PE' # + 'D'
inside = 'EP' # + 'D'
outside = 'EP' # + 'D'

# total_am = df_no_g[amidation_selector].shape[0]
high_scoring = 0.01 < df_no_g["Predictions", "cv_f_logreg"] 
total_am = df_no_g[amidation_selector & high_scoring].shape[0]

total_not_am = df_no_g[~amidation_selector].shape[0]

with open('figures/paper/7b.txt', 'a') as f:
    f.write('Top\n')
    def _print(*args):
        print(*args)
        print(*args, file=f)
    for _in in inside:
        for _out in outside:
            e_inside_selector = df_no_g["Annotations", "C Inside"].apply(lambda x: x[3] == _in)
            p_outside_selector = df_no_g["Annotations", "C Flanking"].apply(lambda x: x[0] == _out)

    #         am_io = df_no_g.loc[amidation_selector & e_inside_selector & p_outside_selector].shape[0] / total_am
    #         am_i = df_no_g.loc[amidation_selector & e_inside_selector].shape[0] / total_am
    #         am_o = df_no_g.loc[amidation_selector & p_outside_selector].shape[0] / total_am
            am_io = df_no_g.loc[high_scoring & amidation_selector & e_inside_selector & p_outside_selector].shape[0] / total_am
            am_i = df_no_g.loc[high_scoring & amidation_selector & e_inside_selector].shape[0] / total_am
            am_o = df_no_g.loc[high_scoring & amidation_selector & p_outside_selector].shape[0] / total_am

            not_am_io = df_no_g.loc[~amidation_selector & e_inside_selector & p_outside_selector].shape[0]  / total_not_am
            not_am_i = df_no_g.loc[~amidation_selector & e_inside_selector].shape[0] / total_not_am
            not_am_o = df_no_g.loc[~amidation_selector & p_outside_selector].shape[0]  / total_not_am

            _print(f'{_in}|*: am={100 * am_i:4.2f}% bg={100 * not_am_i:4.2f}%, enrichment={am_i / not_am_i:4.2f}')
            _print(f'*|{_out}: am={100 * am_o:4.2f}% bg={100 * not_am_o:4.2f}%, enrichment={am_o / not_am_o:4.2f}')
            _print(f'{_in}|{_out}: am={100 * am_io:4.2f}% bg={100 * not_am_io:4.2f}%, enrichment={am_io / not_am_io:4.2f}')
            both = am_io / not_am_io
            expected_both = am_i * am_o / (not_am_i * not_am_o)
            _print(f'---: expected={expected_both:4.2f} observed={both:4.2f}, extra_enrichment={both / expected_both:4.2f}')
            _print()


```

## Figure PPV vs. PeptideRanker


```python
results_dir = 'nested_cv'
df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
add_test_probabilities(df, 'cv_f_logreg', results_dir )
```


```python
from scipy.stats import spearmanr

data = df.loc[df['Annotations', 'Known']]
x = data[('Predictions', 'PeptideRanker')]
y = data[('Predictions', 'cv_f_logreg')]
sns.regplot(x=x, y =y)
spearmanr(x,y)

```
```python
results_dir = 'nested_cv'
df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
add_test_probabilities(df, 'cv_f_logreg', results_dir )
obs_data = df

df = pd.read_pickle('features/mouse_features_paper_assembly_sklearn_with_peptideranker.pickle')
results_dir = 'nested_cv_assembly'
add_test_probabilities(df, 'cv_f_logreg', results_dir )
ppv_data = df
```


```python
# uniprot_secreted = pd.read_csv('https://www.uniprot.org/uniprot/?query=locations:(location:%22Secreted%20[SL-0243]%22)&fil=organism%3A%22Mus+musculus+(Mouse)+[10090]%22&format=tab#', sep='\t')
is_secreted = set(uniprot_secreted["Entry"])
def create_secreted(index):
    secreted = pd.Series(False, index=index, dtype=bool)
    for campaign_id, protein_id, start ,stop in index:
    #for campaign_id, pep_id in index.peptidomics.iter_index():
        _index = campaign_id, protein_id, start, stop
        secreted[_index] = protein_id in is_secreted
    return secreted
    
obs_data["Annotations", "Secreted"] = create_secreted(obs_data.index)
ppv_data["Annotations", "Secreted"] = create_secreted(obs_data.index)
```


```python
def create_known(file_name):
    rename_columns = {'begin': 'start', 'end': 'stop', 'seq': "Sequence", "full_name": "Full Name", "short_names": "Short Names",
                      'type': "Type"}
    known = pd.read_csv(file_name, sep='\t')
    known = known.loc[(known["type"] == 'peptide') | (known["type"] == 'propeptide')].copy()
    known.rename(columns = rename_columns, inplace = True)
    known.set_index(['protein_id', 'start', 'stop'], inplace=True)
    return known[["Type", "Full Name", "Short Names"]]
    
known_2017 = create_known("uniprot/test_data_10090_known.tsv")
known_2020 = create_known("uniprot/2020_04_10090_known.tsv")
```


```python
obs_annotations = obs_data["Annotations"]
ppv_annotations = ppv_data["Annotations"]

obs_annotations = obs_annotations.join(known_2017[["Type", "Full Name", "Short Names"]],
                           on=['protein_id', 'start', 'stop'])

#obs_annotations.join(known_2017["Type"], on=['protein_id', 'start', 'stop'])

for peptide_anno in ["Type", "Full Name", "Short Names"]:
    if peptide_anno in obs_annotations:
        del obs_annotations[peptide_anno]
    obs_data["Annotations", peptide_anno] = obs_annotations.join(known_2017[peptide_anno], on=['protein_id', 'start', 'stop'])[peptide_anno]
    ppv_data["Annotations", peptide_anno] = ppv_annotations.join(known_2017[peptide_anno], on=['protein_id', 'start', 'stop'])[peptide_anno]


propeptide_selector = obs_data["Annotations", "Type"] == 'propeptide'
unnamed_selector = obs_data["Annotations", "Full Name"].isna()
obs_data.loc[propeptide_selector & unnamed_selector, ("Annotations", "Full Name")] = "Unamed Propeptide"

propeptide_selector = ppv_data["Annotations", "Type"] == 'propeptide'
unnamed_selector = ppv_data["Annotations", "Full Name"].isna()
ppv_data.loc[propeptide_selector & unnamed_selector, ("Annotations", "Full Name")] = "Unamed Propeptide"
```


```python
# add names
proteome_annotation = pd.read_csv(
    "uniprot/10090_uniprot.protein.tsv",
    sep='\t')
proteome_annotation.set_index("Protein ID", inplace=True)
name_serie = proteome_annotation[proteome_annotation["Name"] == "Full Name"]["Value"]
gene_serie = proteome_annotation[proteome_annotation["Name"] == "Gene Name"]["Value"]

name_dict = dict(zip(list(name_serie.index), name_serie.values))
gene_dict = dict(zip(list(gene_serie.index), gene_serie.values))

extra_names = {}
extra_genes = {}
for protein_id, other_ids in proteome_annotation[proteome_annotation["Name"] == "Other IDs"]["Value"].iteritems():
    for other_id in eval(other_ids):
        if protein_id in name_dict:
            extra_names[other_id] = name_dict[protein_id]
        if protein_id in gene_dict:
            extra_genes[other_id] = gene_dict[protein_id]
name_dict.update(extra_names)
gene_dict.update(extra_genes)

name_serie = pd.Series(name_dict, name='Protein Name')
gene_serie = pd.Series(gene_dict, name='Gene Name')
```


```python
obs_data["Annotations", "Gene Name"] = obs_annotations.join(gene_serie, on='protein_id')["Gene Name"]
obs_data["Annotations", "Protein Name"] = obs_annotations.join(name_serie, on='protein_id')["Protein Name"]
ppv_data["Annotations", "Gene Name"] = ppv_annotations.join(gene_serie, on='protein_id')["Gene Name"]
ppv_data["Annotations", "Protein Name"] = ppv_annotations.join(name_serie, on='protein_id')["Protein Name"]

```


```python
obs_known = obs_data["Annotations"].loc[obs_data["Annotations", "Known"]].copy()
obs_known_uniq = obs_known.reset_index().set_index(['protein_id', 'start', 'stop'])
obs_known_uniq = obs_known_uniq.filter(items=["Type", "Full Name", "Short Names"])

obs_known = obs_data["Annotations"].loc[obs_data["Annotations", "Known"]].copy()
obs_known_uniq = obs_known.reset_index().set_index(['protein_id', 'start', 'stop'])
obs_known_uniq = obs_known_uniq.filter(items=["Type", "Full Name", "Short Names"]).reset_index().drop_duplicates()

ppv_known = ppv_data["Annotations"].loc[ppv_data["Annotations", "Known"]].copy()
ppv_known_uniq = ppv_known.reset_index().set_index(['protein_id', 'start', 'stop'])
ppv_known_uniq = ppv_known_uniq.filter(items=["Type", "Full Name", "Short Names"]).reset_index().drop_duplicates()

build_known = ppv_data["Annotations"].loc[ppv_data["Annotations", "Known"].astype(bool) & (ppv_data["MS Bool", "observed"] == False)].copy()
build_known_uniq = build_known.reset_index().set_index(['protein_id', 'start', 'stop'])
build_known_uniq = build_known_uniq.filter(items=["Type", "Full Name", "Short Names"]).reset_index().drop_duplicates()

with pd.ExcelWriter('figures/supplement/peptidomics_known3.xlsx') as writer:  
    obs_known.to_excel(writer, sheet_name="Observed Known")
    obs_known_uniq.to_excel(writer, sheet_name="Observed Unique Known")
    ppv_known.to_excel(writer, sheet_name="Build Known")
    ppv_known_uniq.to_excel(writer, sheet_name="Build Unique Known")
    build_known_uniq.to_excel(writer, sheet_name="Only Build Unique Known")
print(ppv_known_uniq.shape)
```


```python
# Add an "All Tissues" column
import collections
from tqdm import tqdm_notebook

def to_odds(p):
    return p / (1 - p)

def to_prob(o):
    return o / (o + 1)

# def join_posterior(prob, prior):
#     like_odds = (np.array(prob) / prior).prod()
#     post_odds = like_odds * to_odds(prior)
#     return to_prob(post_odds)

def join_posterior(prob, prior):
    return 1 - (1-np.array(prob)).prod()

obs_prior = obs_data["Annotations", "Known"].astype(bool).sum() / obs_data.shape[0]
tmp_data = collections.defaultdict(list)
all_tissues = pd.Series(name=("Annotations", "All Tissues"), index=obs_data.index, dtype=str)
joined_posterior_prob = pd.Series(name=("Annotations", "Joined Tissue Posterior"), index=obs_data.index, dtype=str)
predictions_sorted = obs_data["Predictions", "cv_f_logreg"].sort_values()[::-1]
for (tissue, protein, start, stop), value in predictions_sorted.iteritems():
    tmp_data[protein, start, stop].append((tissue, value))
for (protein_id, start, stop), tissues_and_probs in tqdm_notebook(tmp_data.items(), desc="Joining Predictions"):
    tissues, probs = list(zip(*tissues_and_probs))
    p = join_posterior(probs, obs_prior)
    letters = (','.join([t.split(' ')[1][0] for t in tissues])).upper()
    for tissue in tissues:
        all_tissues[tissue, protein_id, start, stop] = letters
        joined_posterior_prob[tissue, protein_id, start, stop] = p
        
obs_data["Annotations", "All Tissues"] = all_tissues
obs_data["Annotations", "Joined Tissue Score"] = joined_posterior_prob
```

```python
# Warning this code takes hours to run, so skip this unless you really need it.

ppv_prior = ppv_data["Annotations", "Known"].astype(bool).sum() / ppv_data.shape[0]
ppv_tmp_data = collections.defaultdict(list)
ppv_all_tissues = pd.Series(name=("Annotations", "All Tissues"), index=ppv_data.index, dtype=str)
ppv_joined_posterior_prob = pd.Series(name=("Annotations", "Joined Tissue Posterior"), index=obs_data.index, dtype=str)
ppv_predictions_sorted = ppv_data["Predictions", "cv_f_logreg"].sort_values()[::-1]
for (tissue, protein, start, stop), value in ppv_predictions_sorted.iteritems():
    ppv_tmp_data[protein, start, stop].append((tissue, value))
for (protein_id, start, stop), tissues_and_probs in tqdm_notebook(ppv_tmp_data.items(), desc="Joining Predictions"):
    tissues, probs = list(zip(*tissues_and_probs))
    p = join_posterior(probs, ppv_prior)
    letters = (','.join([t.split(' ')[1][0] for t in tissues])).upper()
    for tissue in tissues:
        ppv_all_tissues[tissue, protein_id, start, stop] = letters
        ppv_joined_posterior_prob[tissue, protein_id, start, stop] = p
        
ppv_data["Annotations", "All Tissues"] = ppv_all_tissues
ppv_data["Annotations", "Joined Tissue Score"] = ppv_joined_posterior_prob
        
        
```

```python
obs_data = pd.read_pickle('tables/obs_data_export_tempsave.pkl')
ppv_data = pd.read_pickle('tables/ppv_data_export_tempsave.pkl')
```


```python
def to_yes_no(selector):
    series = pd.Series("No", index=selector.index)
    series[selector] = "Yes"
    return series

import re
```


```python
obs_data['Annotations', 'Prediction'] = obs_data['Predictions']['cv_f_logreg']
ppv_data['Annotations', 'Prediction'] = ppv_data['Predictions']['cv_f_logreg']
```


```python
# Warning supplement 7 can take 1 day to create!
ppv_cutoff = 0.01
supplements = ((obs_data, 'figures/supplement/supplement_6.xlsx'),
               (ppv_data, 'figures/supplement/supplement_7.xlsx'))

for (data, file_path) in supplements:
    with pd.ExcelWriter(file_path) as writer:  
        df_out = data.sort_values(by=("Annotations", "Prediction"))[::-1]

        tissue_indexes = [slice(None)] + list(data.index.levels[0])
        tissue_names = ['All Tissues'] + [t.split()[1] for t in tissue_indexes[1:]]

        for i, (tissue_index, tissue_name) in enumerate(zip(tissue_indexes, tissue_names)):
            df_sheet = pd.DataFrame()
            df_tissue = df_out.loc[tissue_index]
            
            if tissue_name == "All Tissues Combined":
                pred_str = "Joined Tissue Score"
            df_tissue = df_tissue[ppv_cutoff < df_tissue["Annotations", "Prediction"]]
            df_tissue_an = df_tissue["Annotations"]
            
            df_sheet["PPV Score"] = df_tissue_an["Prediction"]
            # df_sheet["PeptideRanker Score"] = df_tissue['Predictions', 'PeptideRanker']
            if tissue_name.startswith("All Tissues"):
                df_sheet["Peptide ID"] = ['_'.join(map(str, i)) for i in df_tissue.index.droplevel(0)]
            else:
                df_sheet["Peptide ID"] = ['_'.join(map(str, i)) for i in df_tissue.index]
            known_selector = df_tissue_an["Known"].astype(bool)
            df_sheet["Gene Name"] = df_tissue_an["Gene Name"]
            df_sheet["Protein Name"] = df_tissue_an["Protein Name"]
            df_sheet["Peptide Name"] = "Novel"
            df_sheet.loc[known_selector, "Peptide Name"] = df_tissue_an.loc[known_selector, "Full Name"]
            df_sheet["N Flank"] = df_tissue_an["N Flanking"]
            df_sheet["% Acetylation"] = df_tissue["MS Frequency", "acetylation"].round(3)
            df_sheet["Sequence"] = df_tissue_an["Sequence"]
            df_sheet["% Amidation"] = df_tissue["MS Frequency", "amidation"].round(3)
            df_sheet["C Flank"] = df_tissue_an["C Flanking"]
            df_sheet["Secreted"] = to_yes_no(df_tissue_an["Secreted"].astype(bool))
            if tissue_name == "All Tissues":
                df_sheet["Tissue"] = [i[0].split(' ')[1] for i in df_tissue.index]
            df_sheet["All Tissues"] = df_tissue_an["All Tissues"]
            # df_sheet["PPV Score Combined"] = df_tissue_an["All Tissues Combined"]
            # df_sheet["PPV Score Combined"] = df_tissue_an["Joined Tissue Score"]

            df_sheet["Motif N"] = df_sheet["N Flank"].apply(lambda f: re.search('..[KR][KR]|K..K|R..R', f) is not None)
            df_sheet["Motif C"] = df_sheet["C Flank"].apply(lambda f: re.search('G[KR][KR].|[KR][KR]..|K..K|R..R', f) is not None)
            df_sheet["Split"] = df_tissue_an['Fold']
    #         df_sheet_out = df_sheet[df_sheet["PPV Score"] > ppv_cutoff]
            df_sheet.to_excel(writer, sheet_name=tissue_name, index=False)
            if ("MS Bool", "observed") in data.columns:
                df_sheet["Build"] = df_tissue["MS Bool", "observed"].apply(lambda b: 'Observed' if b else 'Assembled')
                
                
        # combined scores
        tissue_indexes = [slice(None)] + list(data.index.levels[0])
        df_out = data.sort_values(by=[("Annotations", "Joined Tissue Score"), ("Annotations", "Prediction")])[::-1].droplevel(0)
        # df_out = df_out.loc[~out_df.index.duplicated()]
        df_out = df_out.loc[~df_out.index.duplicated()]
        tissue_name = "All Tissues Combined"
        tissue_index = slice(None)
                # 8_50

        df_sheet = pd.DataFrame()
        df_tissue = df_out.loc[tissue_index]
        pred_str = "Prediction"
        if tissue_name == "All Tissues Combined":
            pred_str = "Joined Tissue Score"
        df_tissue = df_tissue[ppv_cutoff < df_tissue["Annotations", pred_str]]
        df_tissue_an = df_tissue["Annotations"]
        df_sheet["PPV Score Combined"] = df_tissue_an["Joined Tissue Score"] #df_tissue_an["All Tissues Combined"]
        df_sheet["Peptide ID"] = ['_'.join(map(str, i)) for i in df_tissue.index]#.droplevel(0)]
        known_selector = df_tissue_an["Known"].astype(bool)
        df_sheet["Gene Name"] = df_tissue_an["Gene Name"]
        df_sheet["Protein Name"] = df_tissue_an["Protein Name"]
        df_sheet["Peptide Name"] = "Novel"
        df_sheet.loc[known_selector, "Peptide Name"] = df_tissue_an.loc[known_selector, "Full Name"]
        df_sheet["N Flank"] = df_tissue_an["N Flanking"]
        df_sheet["% Acetylation"] = df_tissue["MS Frequency", "acetylation"].round(3)
        df_sheet["Sequence"] = df_tissue_an["Sequence"]
        df_sheet["% Amidation"] = df_tissue["MS Frequency", "amidation"].round(3)
        df_sheet["C Flank"] = df_tissue_an["C Flanking"]
        df_sheet["Secreted"] = to_yes_no(df_tissue_an["Secreted"].astype(bool))
        df_sheet["All Tissues"] = df_tissue_an["All Tissues"]
        df_sheet["PPV Score"] = df_tissue_an["Prediction"]
        # df_sheet["PeptideRanker Score"] = df_tissue['Predictions', 'PeptideRanker']
        df_sheet["Motif N"] = df_sheet["N Flank"].apply(lambda f: re.search('..[KR][KR]|K..K|R..R', f) is not None)
        df_sheet["Motif C"] = df_sheet["C Flank"].apply(lambda f: re.search('G[KR][KR].|[KR][KR]..|K..K|R..R', f) is not None)
        df_sheet["Split"] = df_tissue_an['Fold']
#         df_sheet_out = df_sheet[df_sheet["PPV Score"] > ppv_cutoff]
        df_sheet.to_excel(writer, sheet_name=tissue_name, index=False)

```


## PeptideRanker comparison


```python
sns.set_style(style='white')
tissue_indexes = [slice(None)] + list(obs_data.index.levels[0])
df_out = obs_data.sort_values(by=[("Annotations", "Joined Tissue Score"), ("Predictions", "cv_f_logreg")])[::-1].droplevel(0)
# df_out = df_out.loc[~out_df.index.duplicated()]
df_out = df_out.loc[~df_out.index.duplicated()]
tissue_name = "All Tissues Combined"
tissue_index = slice(None)
        # 8_50

df_tissue = df_out.loc[tissue_index]
df_tissue["Annotations", "Joined Tissue Score"] = df_tissue["Annotations", "Joined Tissue Score"].astype(float)


plt.figure(figsize=(15,10))

data = df_tissue.nlargest(300,('Annotations', 'Joined Tissue Score'))
y = data[('Predictions', 'PeptideRanker')]
x = data[('Annotations', 'Joined Tissue Score')]
hue = data[('Annotations', 'Known')]
sns.scatterplot(x=x, y =y, hue=hue, alpha=0.5, legend=None)

for idx, row in data.iterrows():
    name = ':'.join([str(x) for x in row.name])
    if row['Annotations', 'Known']:
        name =  row['Annotations', 'Full Name']

    if row[('Annotations', 'Joined Tissue Score')]> 0.2:
        plt.text(row[('Annotations', 'Joined Tissue Score')], row[('Predictions', 'PeptideRanker')], name, size='small')
    elif row[('Predictions', 'PeptideRanker')] > 0.4:
        plt.text(row[('Annotations', 'Joined Tissue Score')], row[('Predictions', 'PeptideRanker')], name, size='small')


plt.savefig("figures/supplement/ppv_vs_peptideranker.png")
plt.savefig("figures/supplement/ppv_vs_peptideranker.svg")
```
```python

```

```python
# proteome_secreted = len(uniprot_proteome_secreted)/len(uniprot_proteome) *100
# peptidome_secreted = obs_data['Annotations', 'Secreted'].sum() / len(obs_data) *100

# sub_df = obs_data.loc[obs_data['Predictions', 'cv_f_logreg']>0.01]
# high_secreted = sub_df['Annotations', 'Secreted'].sum() / len(sub_df) *100


# sub_df = obs_data.nlargest(200, ('Predictions', 'cv_f_logreg'))
# top_secreted = sub_df['Annotations', 'Secreted'].sum() / len(sub_df) *100


# pd.Series({'Prot.': proteome_secreted, 'Pept.': peptidome_secreted, 'PPV>\n0.01': high_secreted, 'PPV\nTop200':top_secreted}).plot(kind='bar', rot=0)

```
```python
tissue_indexes = [slice(None)] + list(obs_data.index.levels[0])
df_out = obs_data.sort_values(by=[("Annotations", "Joined Tissue Score"), ("Predictions", "cv_f_logreg")])[::-1].droplevel(0)
# df_out = df_out.loc[~out_df.index.duplicated()]
df_out = df_out.loc[~df_out.index.duplicated()]
tissue_name = "All Tissues Combined"
tissue_index = slice(None)
        # 8_50

df_tissue = df_out.loc[tissue_index]
df_tissue["Annotations", "Joined Tissue Score"] = df_tissue["Annotations", "Joined Tissue Score"].astype(float)


plt.figure(figsize=(15,10))

data = df_tissue.nlargest(300,('Annotations', 'Joined Tissue Score'))
y = data[('Predictions', 'PeptideRanker')]
x = data[('Annotations', 'Joined Tissue Score')]
hue = data[('Annotations', 'Known')]
sns.scatterplot(x=x, y =y, hue=hue, alpha=0.5, legend=None)

for idx, row in data.iterrows():
    name = ':'.join([str(x) for x in row.name])
    if row['Annotations', 'Known']:
        name =  row['Annotations', 'Full Name']

    if row[('Annotations', 'Joined Tissue Score')]> 0.2:
        plt.text(row[('Annotations', 'Joined Tissue Score')], row[('Predictions', 'PeptideRanker')], name, size='small')
    elif row[('Predictions', 'PeptideRanker')] > 0.4:
        plt.text(row[('Annotations', 'Joined Tissue Score')], row[('Predictions', 'PeptideRanker')], name, size='small')
```

```python
g.figure.savefig("figures/supplement/ppv_vs_peptideranker.png")
g.figure.savefig("figures/supplement/ppv_vs_peptideranker.svg")
```

## Linkout for supplemental tables
Thise were made after "prettifying" the tables and are thus here made as a small tables that is attached to the other table via vlookup in Excel

```python
df = pd.read_pickle('features/mouse_features_paper_sklearn_with_peptideranker.pickle')
df.head(2)
```

```python
def get_eggnog_mapper(from_manog_file, to_manog_file):
    nog_to_proteins = {}
    mapper = {}
    with open(from_manog_file) as from_file, open(to_manog_file) as to_file:
        # from is mouse
        # from protein to nog
        for line in from_file:
            nog, best_id, other_ids, *mapping_scores = line.split('\t')
            nog_to_proteins[nog] = {best_id}
            nog_to_proteins[nog] = (set(other_ids.split(',')) - {''}) | {best_id}
        
        # to is human
        # from proteins to proteins via nog
        for line in to_file:
            nog, to_protein, other_ids, *mapping_scores = line.split('\t')
            if nog not in nog_to_proteins:
                continue  # no ortholog
            for from_protein in nog_to_proteins[nog]:
                mapper[from_protein] = to_protein
    return mapper
mouse_to_human_mapper = get_eggnog_mapper('mapping/maNOG_10090_nogmap.tsv', 'mapping/maNOG_9606_nogmap.tsv')
```

```python
index = df.index.get_level_values('protein_id').unique()
```

```python
human_series = pd.Series({i: mouse_to_human_mapper.get(i, '') for i in index}, name="Human Protein ID")

pharos_series = pd.Series({i: f"https://pharos.nih.gov/targets/{mouse_to_human_mapper.get(i, 'DELETE_ME')}" for i in index}, name="Pharos")
pharos_series.loc[pharos_series.str.endswith('DELETE_ME')] = ""
pharos_series["A2A5R2"]  # sanity check
```

```python
diseases_series = pd.Series({i: f"http://amigo.geneontology.org/amigo/gene_product/UniProtKB:{mouse_to_human_mapper.get(i, 'DELETE_ME')}" for i in index}, name="DISEASES")
diseases_series.loc[diseases_series.str.endswith('DELETE_ME')] = ""
diseases_series["A2A5R2"]  # sanity check
```

```python
diseases_series = pd.Series({i: f"https://diseases.jensenlab.org/Search?query={mouse_to_human_mapper.get(i, 'DELETE_ME')}" for i in index}, name="DISEASES")
diseases_series.loc[diseases_series.str.endswith('DELETE_ME')] = ""
diseases_series["A2A5R2"]  # sanity check
```

```python
go_mouse_series = pd.Series({i: f'http://amigo.geneontology.org/amigo/gene_product/UniProtKB:{i}' for i in index}, name="GO Mouse")
go_human_series = pd.Series({i: f"http://amigo.geneontology.org/amigo/gene_product/UniProtKB:{mouse_to_human_mapper.get(i, 'DELETE_ME')}" for i in index}, name="GO Human")
go_human_series.loc[go_human_series.str.endswith('DELETE_ME')] = ""

```

```python
df_link = pd.DataFrame({s.name: s for s in (human_series, pharos_series, diseases_series, go_mouse_series, go_human_series)})
```

```python
df_link.to_excel("figures/supplement/linkout.xlsx")
```
