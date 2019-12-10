# core imports
import pickle

# 3rd party imports
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#  import seaborn as sns
import pymc3 as pm
import scipy as sp
#  import theano.tensor as tt

#  from scipy.stats import beta
#  from scipy.special import expit
#  from matplotlib import gridspec

# lodal imports
import ppv  # noqa
#  from scripts.run import create_features


# setup
mpl.use('agg')
plt.style.use('seaborn-white')
color = '#87ceeb'
f_dict = {'size': 16}


#  def qr_data(data):
#      # center data
#      X = data.ppv._transform().ppv.predictors
#      meanx = X.mean().values
#      scalex = X.std().values
#      zX = ((X - meanx) / scalex).values


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

        pm.model_to_graphviz(model)
    return model


def create_model_qr(zX, y, graph=False):

    with pm.Model() as model:
        # qr decompose predictors and normalize
        Q, R = sp.linalg.qr(zX, mode='economic')
        Q *= zX.shape[0]
        R /= zX.shape[0]

        zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
        zbetaj = pm.Normal('zbetaj', mu=0, sd=2, shape=(zX.shape[1]))  # noqa

        betaj_tilde = pm.Deterministic('betaj_tilde', pm.math.dot(R, zbetaj))
        p = pm.invlogit(zbeta0 + pm.math.dot(Q, betaj_tilde))

        likelihood = pm.Bernoulli('likelihood', p, observed=y.values)  # noqa

        pm.model_to_graphviz(model)
    return model


def sample(model, n=3000):
    with model:
        trace = pm.sample(n, chains=8, cores=1)
    return trace


def sample_model(zX, y, down_sample=None, model_type='default'):
    #  positives = df[df["Annotations", "Known"]]
    #  negatives = df[~df["Annotations", "Known"].astype(bool)]
    #  data = pd.concat((positives, negatives))

    if model_type == 'default':
        model = create_model(zX, y)
    else:
        model = create_model_qr(zX, y)
    with model:
        #  trace = pm.sample(1000, cores=8)
        trace = pm.sample(3000, cores=4, chains=8)
    if down_sample is None:
        pickle.dump(trace, open("pickle/model_trace_obs_data.pickle", "wb"))
    else:
        pickle.dump(trace, open("pickle/model_trace_obs_data_{}.pickle".format(down_sample), "wb"))
        pm.traceplot(trace)
    pm.traceplot(trace)
    fig = plt.gcf()
    fig.savefig("plots/trace_{}_{}.pdf".format(down_sample, model_type))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="number of processes", default=16, type=int)
    parser.add_argument("-s", help="class balance after down-sampeling", default=None, type=int)
    parser.add_argument("--model", help="model type", default='default')
    parser.add_argument("--drop-weak-features", action='store_true', default=False)
    parser.add_argument("--build", action='store_true', default=False)
    parser.add_argument("--load", action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    df = pickle.load(open("pickle/mouse_features.pickle", 'rb'))
    # only train on onserved
    if args.build:
        raise NotImplementedError("TODO!!!")
        base_name = "ppv_data"
    else:
        observed_selector = df["MS Frequency", "observed"] != 0
        o_positives = df[observed_selector].ppv.positives
        o_negatives = df[observed_selector].ppv.negatives
        data = pd.concat((o_positives, o_negatives))
        base_name = "obs_data"

    # reduce stuff
    down_sample = args.s
    if args.s is not None:
        base_name += '_{}'.format(args.s)
        negative_samples = down_sample * o_positives.shape[0]
        data = pd.concat((o_positives, o_negatives.sample(negative_samples)))

    if args.drop_weak_features:
        base_name += '_strong'
        data = data.ppv._drop_weak_features()
    data = data.ppv._transform()

    if args.load:
        trace, meanx, scalex = pickle.load(
            open("pickle/parameter_trace_{}.pickle".format(base_name), 'rb'))
    else:
        X = data.ppv.predictors
        meanx = X.mean()
        scalex = X.std()
        #  zX = ((X - meanx) / scalex).values
        model = data.ppv.create_model(meanx, scalex)
        with model:
            #  trace = pm.sample(500, cores=4, chains=8)
            trace = pm.sample(3000, cores=4, chains=8)

        pickle.dump(
            (trace, meanx, scalex, np.nanmedian(df.values.flatten())),
            open("pickle/parameter_trace_{}.pickle".format(base_name), 'wb'))

    # posteori
    data.ppv.plot_posterior(
        trace, save_path="figures/posterior_z_{}.pdf".format(base_name))
    data.ppv.plot_posterior(
        trace, meanx, scalex,
        save_path="figures/posterior_{}.pdf".format(base_name))
