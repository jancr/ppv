# core imports
import pickle
import math

# 3rd party imports
import numpy as np
import scipy as sp
import pandas as pd
import pymc3 as pm
from matplotlib import pyplot as plt


# kruschke style plots
plt.style.use('seaborn-white')
color = '#87ceeb'
f_dict = {'size': 16}


class PPVModel:
    def __init__(self, df, mini_batch=0):
        self.predictors = df.ppv.predictors
        self.target = df.ppv.target
        if not isinstance(mini_batch, int) or mini_batch < 0:
            raise ValueError("mini_batch must be a positive integer, not {}".format(mini_batch))
        self.mini_batch = mini_batch

        # scale data
        self.meanx = self.predictors.mean()
        self.scalex = self.predictors.std()
        self.model = self._create_model()

        # refrence data info
        data = df["Annotations", "Intensity"].values.flatten()
        data = np.log10(data[~np.isnan(data)])
        self.df_median = np.median(data)
        self.df_scale = data.std()

        # infered from trace
        self.trace = self.intercept = self.parameters = None

    def _create_model(self):
        zX = ((self.predictors - self.meanx) / self.scalex).values
        y = self.target.astype(int).values
        shape = zX.shape
        if self.mini_batch:
            zX = pm.Minibatch(zX, batch_size=self.mini_batch)
            y = pm.Minibatch(y, batch_size=self.mini_batch)
        with pm.Model() as model:
            zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
            zbetaj = pm.Normal('zbetaj', mu=0, sd=2, shape=shape[1])

            p = pm.invlogit(zbeta0 + pm.math.dot(zX, zbetaj))
            likelihood = pm.Bernoulli('likelihood', p, observed=y, total_size=shape[0])  # noqa
            #  pm.model_to_graphviz(model)
        return model

    @classmethod
    def load(cls, path, true_prior=None):
        self = pickle.load(open(path, 'rb'))
        if true_prior:
            # old versions only had 1 intercept
            if not hasattr(self, 'data_intercept'):
                self.data_intercept = self.intercept
            self.intercept = self.get_real_intercept(true_prior)
        return self

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    def get_real_intercept(self, true_prior):
        data_prior = self.target.astype(bool).sum() / self.predictors.shape[0]
        return self.data_intercept - math.log(data_prior) + math.log(true_prior)

    def add_trace(self, trace, true_prior=None):
        self.trace = trace
        self.intercept = self.data_intercept = self.beta0.mean()
        if true_prior:
            self.intercept = self.get_real_intercept(true_prior)
        self.parameters = pd.Series(self.betaj.mean(axis=0), index=self.scalex.index)

    def rescale_upf_data_to_refrence(self, df):
        data = df.values.flatten()
        data = np.log10(data[~np.isnan(data)])
        df_median = np.median(data)
        df_scale = data.std()
        df_zscaled = (np.log10(df) - df_median) / df_scale
        df_rescaled = (df_zscaled * self.df_scale) + self.df_median
        return 10 ** df_rescaled

    def predict(self, df, transform=True):
        if transform:
            df = df.ppv._transform()
        predictors = df.ppv.predictors.filter([x for x in self.columns], axis=1)
        ods = pd.Series(self.intercept + predictors.dot(self.parameters), dtype=float)
        return sp.special.expit(ods)

    ############################################################
    # plots
    ############################################################
    @classmethod
    def _predictor_canvas(cls, predictors, axes_size=4, extra=0, shape=None):
        features = predictors.shape[-1] + extra
        if shape is None:
            n_col = math.ceil(features ** 0.5)
            n_row = math.ceil(features / n_col)
        else:
            n_row, n_col = shape
        fig = plt.figure(figsize=(n_col * axes_size, n_row * axes_size))
        axes = fig.subplots(n_row, n_col)
        for ax in axes.flatten()[features:]:
            ax.axis('off')
        return fig, axes.reshape(n_row, n_col)

    def plot_posterior(self, save_path=None, axes_size=4):
        self._plot_posterior(self.beta0, self.betaj, save_path, axes_size)

    def plot_zposterior(self, save_path=None, axes_size=4):
        self._plot_posterior(self.zbeta0, self.zbetaj, save_path, axes_size)

    def _plot_posterior(self, beta0, betaj, save_path=None, axes_size=4, shape=None):
        fig, axes = self._predictor_canvas(self.predictors, axes_size, 1, shape=shape)
        pm.plot_posterior(beta0.values, point_estimate='mode', ax=axes[0, 0], color=color)
        axes[0, 0].set_xlabel(r'$\beta_0$ (Intercept)', fontdict=f_dict)
        axes[0, 0].set_title('', fontdict=f_dict)
        columns = self.predictors.columns
        for i, (ax, feature) in enumerate(zip(axes.flatten()[1:], columns)):
            pm.plot_posterior(betaj[feature].values, point_estimate='mode',
                              ax=ax, color=color)
            ax.set_title('', fontdict=f_dict)
            ax.set_xlabel(r'$\beta_{{{}}}$ ({})'.format(i + 1, ' '.join(feature)),
                          fontdict=f_dict)
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    ############################################################
    # properties
    ############################################################
    @property
    def columns(self):
        return self.meanx.index

    @property
    def zbeta0(self):
        return pd.Series(self.trace['zbeta0'], name=("Intercept", "Intercept"))

    @property
    def beta0(self):
        offset = np.sum(self.zbetaj * self.meanx / self.scalex, axis=1)
        return pd.Series(self.zbeta0 - offset, name=("Intercept", "Intercept"))

    @property
    def zbetaj(self):
        return pd.DataFrame(self.trace['zbetaj'], columns=self.columns)

    @property
    def betaj(self):
        return self.zbetaj / self.scalex

    #          beta0 = trace['zbeta0'] - np.sum(trace['zbetaj'] * meanx / scalex, axis=1)
    #          betaj = (trace['zbetaj'] / scalex)
    #      else:
    #          beta0 = trace['zbeta0'] - np.sum(trace['zbetaj'], axis=1)
    #          betaj = (trace['zbetaj'])
