
# core imports
import pickle

# 3rd party imports
import pandas as pd
#  import numpy as np
import matplotlib.pyplot as plt
#  import seaborn as sns
import pymc3 as pm
#  import theano.tensor as tt

#  from scipy.stats import beta
#  from scipy.special import expit
#  from matplotlib import gridspec

# lodal imports
import ppv  # noqa
from scripts.run import create_features


# setup
plt.style.use('seaborn-white')
color = '#87ceeb'
f_dict = {'size': 16}


def zscore_data(data):
    X = data.ppv._transform().ppv.predictors
    #  y = data.ppv.target

    meanx = X.mean().values
    scalex = X.std().values
    zX = ((X - meanx) / scalex).values
    return zX


def create_model(zX, y, graph=False):
    with pm.Model() as model:

        zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
        zbetaj = pm.Normal('zbetaj', mu=0, sd=2, shape=(zX.shape[1]))

        p = pm.invlogit(zbeta0 + pm.math.dot(zbetaj, zX.T))

        likelihood = pm.Bernoulli('likelihood', p, observed=y.values)  # noqa

    if graph:
        pm.model_to_graphviz(model)
    return model


def sample(model, n=3000):
    with model:
        trace = pm.sample(n, cores=8)
    return trace


if __name__ == '__main__':
    df = create_features(True, 16)
    positives = df[df["Annotations", "Known"]]
    negatives = df[~df["Annotations", "Known"].astype(bool)]
    data = pd.concat((positives, negatives))

    observed_selector = df["MS Frequency", "observed"] != 0
    o_positives = df[observed_selector].ppv.positives
    o_negatives = df[observed_selector].ppv.negatives
    obs_data = pd.concat((o_positives, o_negatives))
    #  obs_data_balance = pd.concat((o_positives, o_negatives.sample(o_positives.shape[0])))
    obs_data_reduced = pd.concat((o_positives, o_negatives.sample(o_positives.shape[0] * 50)))

    zX = zscore_data(obs_data)
    model = create_model(zX, obs_data.ppv.target)
    with model:
        #  trace = pm.sample(1000, cores=8)
        trace = pm.sample(3000, chains=8)
    pickle.dump(trace, open("pickle/model_trace.pickle", "wb"))
