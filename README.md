
## Instalation 
assumes [pyenv](https://github.com/pyenv/pyenv) is installed

**Note:** During development `anaconda3-2019.03`, but it should also work with python 3.9.

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

lets go back to the `ppv-project` folder and clone this repo

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


## Use the Software

There are two use cases for this project

1) use our model to make predictions for your own data
2) train your own model on your (and our?) data

In either case you need to extract features from your data. Before you can train or predict, so
let's do that


### Extract features

All the features can be found in `ppv-data/features/mouse_features_paper.pickle.gz`

