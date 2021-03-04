# PPV - Predicted Peptide Variant
* A Peptide Feature Extraction tool for Mass Spectrometry Data
* A Bayesian Logistic Classifier, learning the features of uniprot annotated peptides.

## Installation 

This guide assumes that [pyenv](https://github.com/pyenv/pyenv) is installed

**Note:** During development Anaconda3 (`anaconda3-2019.03`) was used, but it
should also work with python 3.9.

First let's install and use the anaconda python we used during development:

```
pyenv install anaconda3-2019.03
pyenv global anaconda3-2019.03
```

**Package Install:** 

```
pip install git+ssh://git@github.com/jancr/ppv.git#egg=ppv
```


**Developer Install:** 

Clone the PPV repository:

```
mkdir ppv-project
cd ppv-project
git clone https://github.com/jancr/ppv.git
```

Install the package

```
cd ppv
pip install -e .
```


## Download Data
data from the paper can be found at `https://github.com/jancr/ppv-data`

Lets go back to the `ppv-project` folder and clone this repo

```
cd ..
git clone https://github.com/jancr/ppv-data
```

Then unzip all the files

```
cd models
gunzip *.gz
cd ../features
gunzip mouse_features_paper.pickle.gz
cd ../..
```

Hopefully your `ppv-project` directory now looks like this:

```
$ ls -lh
total 8.0K
drwxrwxr-x 6 jcr jcr 4.0K Mar  3 15:15 ppv
drwxrwxr-x 6 jcr jcr 4.0K Mar  3 15:21 ppv-data
```

## File Types

There are two core file types in this project

**UPF files:** In the `ppv-data/upf` there are two types of files. The `*.upf`
file which contains 1 line per peptide per sample. It had 3 important concepts:

* **Meta Data**: The field `accno` is the sample id to link it to meta data such as "This is Mouse 5"
* **Peptide ID**: the fields `prot_acc`, `pep_start`, `pep_stop` and `pep_mod_seq`
  amounts to the peptide ID, the `pep_mod_seq` allows us to have seperate ID's
  for peptides with different PTMs
* **Abundance**: the field `intensity` is the abundance recorded by the Mass Spectrometer.

**Sample Meta files:** These files contain meta data about the upf file, this
is necessary for defining groups when doing statistical analysis of the data,
in relation to the PPV algorithm the only field that matters is `rs_acc` which
is used to link to the `accno` field in the upf file, and `subject` which is
the mouse id.

If you want to use the algorithm for your own data you have to convert the
output from the MS into this format.

## Extract features

There are two use cases for this project

1) use our model to make predictions for your own data
2) train your own model on your (and our?) data

In either case you need to extract features from your data. Before you can train or predict, so
let's do that

All the features can be found in
`ppv-data/features/mouse_features_paper.pickle.gz`, this file contains all the
features extracted from all the tissue files. In order to understand how this
file was created let's create it for 1 tissue, doing it for all simply amounts
to using a for loop :)

### Example: create feature data frame for Mouse Brain

Import statements:

```
import pandas as pd
import peputils
from peputils.proteome import fasta_to_protein_hash
import ppv
```

Then we link to the files in `ppv-data`:

```
upf_file = 'upf/mouse_brain_combined.upf'
meta_file = 'upf/mouse_brain_combined.sample.meta'
campaign_name = "Mouse Brain"
mouse_fasta = "uniprot/10090_uniprot.fasta"
known_file = "uniprot/known.tsv"
```

Then we now create a upf data frame, we do this using data frame method
`.peptidomics.load_upf_meta`, which is defined in `peputils`:

```
df_raw = pd.DataFrame.peptidomics.load_upf_meta(upf_file, meta_file, campaign_name)
```

We then normalize this dataframe such that all the peptides found across all
samples sum to the same, to correct for different sample loading.

```
df = df_raw.peptidomics.normalize()
```

Now we have a normalized peptidomics dataframe, it looks like this:

```
df.head()
```


![png of df.head() ](figures/df_head.png)

So much like the `.upf` file we have 1 row for each observed peptide and 1 column
for each sample abundance.

The above dataframe is what is needed for feature extraction, to extract
features from the df use the following method:

```
n_cpu = 8
mouse_proteins = fasta_to_protein_hash(mouse_fasta)

dataset_features = df.ppv_feature_extractor.create_feature_df(
    mouse_proteins, n_cpus=n_cpu, known=known_file, peptides='valid')
```

**Note:** The feature extraction code is parallelized such that if
`n_cpu=8`, then it will concurrently extract features from 8 protein backbones,
as some proteins have a much higher number of peptides than others (and the
algorithm scales O(N^2) with the number of peptides in a protein), the progress
bar seem to stall, when there are only the 1-5 proteins with most peptides
left. Be patient my young padowan, the program is not stuck in an infinite
loop, but it may take some hours to finish.

### Using the Model for Prediction

When using the model for prediction, you need two things:

1. Features: Here we will use `ppv-data/features/mouse_features_paper.pickle`,
   which was also used in the paper.
2. A model: Here we will use the model from the paper
   `ppv-data/models/model_paper_obs_data_strong.ppvmodel`

If you have some other data or have trained your own model, this guide should still work

First let's load the features and model

**Note:** The `.ppvmodel` files have been pickled with python 3.7, with pandas
version 0.24.3, and cannot be unpickeled by pandas 1.0 or newer, also on some
Linux distributions Theano has missing libraries, which can also cause pickeling issues.

```
import ppv

model_file = ppv-data/models/model_paper_obs_data_strong.ppvmodel
model = ppv.model.PPVModel.load(model_file)
predictions = model.predict(dataset_features)
```

### Training your own model 

To Train a model on the set of features we used in the paper (this assumes you have the `dataset_features` from one of the above steps), first we subset the `dataset_features` to the strong features used in the paper:

```
data = dataset_features.ppv._drop_weak_features()
```

**Optional Down sampling:** Training the full model takes days, so you may want
to down sample if you are only playing with the tool

First we save the prior (True/All) so we can adjust the models intercept as if
we trained on the full dataset.
```
true_prior = data.ppv.get_prior()  # calcuate true prior before down sampling!
```

let's down sample so we have 10:1 negatives to positives:

```
down_sample = 10
positives = data.ppv.positives
negatives = data.ppv.negatives
negative_samples = down_sample * positives.shape[0]
data = pd.concat((positives, negatives.sample(negative_samples)))
```

**Training:**

Create a Model object from the data:
```
model = PPVModel(data)
```

Draw 7x2000 samples from the model and add them to the model object:

```
with model.model:
	trace = pm.sample(2000, cores=7, chains=7, target_accept=0.9)
model.add_trace(trace, true_prior=true_prior)
```

**Optional:** Save the model

```
model.save("pickle/model_{}.ppvmodel".format(base_name))
```


## Notes on Dependencies

During development two major packages used by the project underwent large
changes `pandas` changed from version 0.x to 1.x which broke `xarray` and other
`pycm3` dependencies, `pycm3` itself also went trough an existential crisis
because the `theano` project which they were build upon were discontinued, All
these issues seems fixed in Python 3.9 where all packages uses pandas 1.x and
where the `pycm3` project has taken charge of `theano`. To make this project
more accessible we may retrain the models using Python 3.9 in the near future
and upload them to the `ppv-data` repository.
