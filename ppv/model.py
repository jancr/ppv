# core imports
import math
import pathlib
import pickle
import tempfile
import warnings
import zipfile

# 3rd party imports
import numpy as np
import scipy as sp
import pandas as pd
import pymc3 as pm
from pymc3.backends.base import MultiTrace
from arviz.data.inference_data import InferenceData
from matplotlib import pyplot as plt

# kruschke style plots
plt.style.use('seaborn-white')
color = '#87ceeb'
f_dict = {'size': 16}


class BasePPVModel:
    def __init__(self, df, true_prior=None, mini_batch=0):
        self.df = df
        self.true_prior = true_prior
        self.predictors = df.ppv.predictors
        self.target = df.ppv.target
        if not isinstance(mini_batch, int) or mini_batch < 0:
            raise ValueError("mini_batch must be a positive integer, not {}".format(mini_batch))
        self.mini_batch = mini_batch

        # scale data
        self.meanx = self.predictors.mean()
        self.scalex = self.predictors.std()
        self.model = self._create_model()

        # inferred from trace
        self.trace = self.intercept = self.parameters = None

    def _prep_data():
        zX = ((self.predictors - self.meanx) / self.scalex).values
        y = self.target.astype(int).values
        if self.mini_batch:
            zX = pm.Minibatch(zX, batch_size=self.mini_batch)
            y = pm.Minibatch(y, batch_size=self.mini_batch)

    @abstracet
    def _create_model(self):
        raise NotImplementedError('I am an abstract method, implement me!')

    @classmethod
    def load(cls, path, tmp_dir=None):
        def load_file(file_name, load_function):
            zf.extract(file_name, str(tmp_dir))
            return load_function(str(tmp_dir_path / file_name))

        def load_pickle(file_name):
            tmp_path = tmp_dir_path / file_name
            zf.extract(file_name, str(tmp_dir))
            with tmp_path.open('rb') as fh:
                return pickle.load(fh)

        if issubclass(path.__class__, cls):
            return self
        if zipfile.is_zipfile(path):
            zf = zipfile.ZipFile(path, 'r')
            with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dir:
                tmp_dir_path = pathlib.Path(tmp_dir)

                # load data
                df = load_file('df.pickle', pd.read_pickle)
                kwargs = load_pickle('args.pickle')
                self = cls(df, **kwargs)

                # load posterior samples
                try:
                    trace = load_pickle('inference_samples.pickle')
                except:
                    try:
                        trace = load_pickle('inference_data.pickle')
                    except:
                        # if pickle fails then we have netcdf in the zip file as a backup
                        data_file = 'inference_data.netcdf'
                        path_folder = pathlib.Path(path).parent
                        zf.extract(data_file, path_folder)
                        data_file = (path_folder / data_file).rename(path_folder / data_file)
                        trace = InferenceData.from_netcdf(data_file)
            self.add_trace(trace, true_prior=kwargs.get('true_prior', None))
        else:
            # if not zip file, then everything has been pickled together:
            self = pickle.load(open(path, 'rb'))
            if true_prior:
                # old versions only had 1 intercept
                if not hasattr(self, 'data_intercept'):
                    self.data_intercept = self.intercept
                self.intercept = self.get_real_intercept(true_prior)
        return self


    def save(self, path: str, inference_data_format: str='pickle', tmp_dir=None):
        """
        Save the object to disk, This is done by creating 3 temporary files based on
        :code:`self.df`, :code:`self.trace` and a code:`dict` with all other fields needed to
        recreate the object.

        :param path: path to file
        :param infrence_data_format: how :code:`self.trace` is saved to disk: 
            
            * pickle or large: the full object is pickled (expect GB file size)
            * netcdf: the full object is saved as a netcdf5 file (expect GB file size)
            * samples or small: only the posterior samples will be pickled (expect MB file size)

        If you expect to use the model with :code:`arviz` to do posterior predictive check or fancy
        plots, then pickle or netcdf should be chosen if you expect to only use it for prediction
        then samples is sufficient

        """

        def save_file(file_name, save_function):
            tmp_path = str(tmp_dir_path / file_name)
            save_function(tmp_path)
            zf.write(tmp_path, file_name)
        def pickle_file(file_name, data):
            tmp_path = tmp_dir_path / file_name
            with tmp_path.open('wb') as fh:
                pickle.dump(data, fh)
            zf.write(str(tmp_path), file_name)
            
        _msg = None
        if self.trace is None:
            _msg = "saving.... obj.trace is None, hint: obj.add_trace(trace)"
            warnings.warn(_msg, RuntimeWarning)
        elif isinstance(self.trace, MultiTrace):
            _msg = "obj.trace is of type MultiTrace, InferenceData should be used instead"
            raise ValueError(_msg)

        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_folder, zipfile.ZipFile(path, 'w') as zf:
            tmp_dir_path = pathlib.Path(tmp_folder)

            save_file('df.pickle', self.df.to_pickle)
            kwargs = {'mini_batch': self.mini_batch, 'true_prior': self.true_prior}
            pickle_file('args.pickle', kwargs)

            if self.trace is None:
                return
            if inference_data_format in ('pickle', 'large'):
                pickle_file('inference_data.pickle', self.trace)
            elif inference_data_format == 'netcdf':
                save_file('inference_data.netcdf', self.trace.to_netcdf)
            elif inference_data_format in ('samples', 'small'):
                #  posterior = {'zbeta0': self.trace.posterior.data_vars['zbeta0'],
                #               'zbetaj': self.trace.posterior.data_vars['zbetaj']}
                posterior = {'zbeta0': self.zbeta0, 'zbetaj': self.zbetaj}
                pickle_file('inference_samples.pickle', posterior)
            else:
                valid_strings = ', '.join(('pickle', 'netcdf', 'samples', 'large', 'small'))
                raise ValueError(f"inference_data_format must be one of: {valid_strings}")



    def get_real_intercept(self, true_prior):
        data_prior = self.target.astype(bool).sum() / self.predictors.shape[0]
        return self.data_intercept - math.log(data_prior) + math.log(true_prior)

    def add_trace(self, trace, true_prior=None):
        if isinstance(trace, MultiTrace):
            _msg = ("trace should preferably be of type InferenceData, "
                    "hint: pm.sample(..., return_inferencedata=True)")
            warnings.warn(_msg, DeprecationWarning)
        self.trace = trace
        self.intercept = self.data_intercept = self.beta0.mean()
        if true_prior:
            self.intercept = self.get_real_intercept(true_prior)
        self.parameters = pd.Series(self.betaj.mean(axis=0), index=self.scalex.index)

    def rescale_upf_data_to_refrence(self, df):
        # reference data info
        ref_data = self.df["Annotations", "Intensity"].values.flatten()
        ref_data = np.log10(ref_data[~np.isnan(ref_data)])
        self.df_median = np.median(ref_data)
        self.df_scale = ref_data.std()


        # new data to be rescaled
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
        for ax in axes.flatten():
            ax.patch.set_visible(False)
            ax.xaxis.grid(False)
        for ax in axes.flatten()[features:]:
            ax.axis('off')
        return fig, axes.reshape(n_row, n_col)

    def _plot_posterior(self, betaj, save_path=None, axes_size=4, shape=None,
                        credible_interval=0.94, color_bad=True, beta0=None):
        fig, axes = self._predictor_canvas(self.predictors, axes_size, 1, shape=shape)
        if beta0 is None:
            beta0 = betaj[0]
            betaj = betaj[1:]
        pm.plot_posterior(beta0.values, point_estimate='mode', ax=axes[0, 0], color=color)
        axes[0, 0].set_xlabel(r'$\beta_0$ (Intercept)', fontdict=f_dict)
        axes[0, 0].set_title('', fontdict=f_dict)
        columns = self.predictors.columns
        for i, (ax, feature) in enumerate(zip(axes.flatten()[1:], columns)):
            pm.plot_posterior(betaj[feature].values, point_estimate='mode', 
                            credible_interval=credible_interval, ax=ax, color=color)
            ax.set_title('', fontdict=f_dict)
            ax.set_xlabel(r'$\beta_{{{}}}$ ({})'.format(i + 1, ' '.join(feature)),
                        fontdict=f_dict)
            if color_bad and not self.is_credible(self.trace['zbetaj'][:, i], credible_interval):
                ax.patch.set_facecolor('#FFCCCB')
                ax.patch.set_visible(True)
        if save_path is not None:
            fig.savefig(save_path, transparent=True)
        return fig

    @classmethod 
    def is_credible(cls, parameter, credible_interval=0.95):
        min_, max_ = pm.stats.hpd(parameter, 1 - credible_interval)
        return not (min_ < 0 < max_)

    ############################################################
    # properties
    ############################################################
    @property
    def columns(self):
        return self.meanx.index

    #  def _extract_vector(self, name):
    #      if isinstance(self.trace, InferenceData):
    #          # arviz has posterior, posterior predictive etc in it's "trace" object
    #          # also it has a shape of (samples, chains, parameters) 
    #          np.asarray(trace.posterior.data_vars['zbetaj']).reshape(-1, shape[-1])
    #  
    #          return self.trace.posterior.data_vars[name]
    #      return self.trace[name]

def PPVModelPaper(BasePPVModel):
    @property
    def zbeta0(self):
        if isinstance(self.trace, InferenceData):
            zbeta0 = np.asarray(self.trace.posterior.data_vars['zbeta0'])
            zbeta0 = zbeta0.reshape(-1)
        elif isinstance(self.trace, (MultiTrace, dict)):
            zbeta0 = self.trace['zbeta0']
        return pd.Series(zbeta0, name=("Intercept", "Intercept"))

    @property
    def beta0(self):
        offset = np.sum(self.zbetaj * self.meanx / self.scalex, axis=1)
        return pd.Series(self.zbeta0 - offset, name=("Intercept", "Intercept"))

    @property
    def zbetaj(self):
        if isinstance(self.trace, InferenceData):
            zbetaj = np.asarray(self.trace.posterior.data_vars['zbetaj'])
            zbetaj = zbetaj.reshape(-1, zbetaj.shape[-1])
        elif isinstance(self.trace, (MultiTrace, dict)):
            zbetaj = self.trace['zbetaj']
        return pd.DataFrame(zbetaj, columns=self.columns)

    @property
    def betaj(self):
        return self.zbetaj / self.scalex


    def _create_model(self):
        zX, y = self._prep_data()
        with pm.Model() as model:
            zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
            zbetaj = pm.Normal('zbetaj', mu=0, sd=2, shape=zX.shape[1])

            p = pm.invlogit(zbeta0 + pm.math.dot(zX, zbetaj))
            likelihood = pm.Bernoulli('likelihood', p, observed=y, total_size=y.shape[0])  # noqa
            #  pm.model_to_graphviz(model)
        return model

    def plot_posterior(self, save_path=None, axes_size=4, shape=None, credible_interval=0.94,
                       color_bad=True):
        return self._plot_posterior(self.betaj, save_path, axes_size, shape,
                                    credible_interval, color_bad, beta0=self.beta0)

    def plot_zposterior(self, save_path=None, axes_size=4, shape=None, credible_interval=0.94,
                        color_bad=True):
        return self._plot_posterior(self.zbetaj, save_path, axes_size, shape,
                                    credible_interval, color_bad, beta0=self.zbeta0)


def PPVModelVector(BasePPVModel):
    """Same as paper model, but intercept is folded into X"""

    def __init__(self, df, true_prior=None, mini_batch=0):
        super(self).__init__()
        self.predictors["Intercept", "Intercept"] = 1
        self.meanx["Intercept", "Intercept"] = 0
        self.scalex["intercept", "Intercept"] = 1

    @property
    def zbetaj(self):
        if isinstance(self.trace, InferenceData):
            zbetaj = np.asarray(self.trace.posterior.data_vars['zbetaj'])
            zbetaj = zbetaj.reshape(-1, zbetaj.shape[-1])
        elif isinstance(self.trace, (MultiTrace, dict)):
            zbetaj = self.trace['zbetaj']
        return pd.DataFrame(zbetaj, columns=self.columns)

    @property
    def betaj(self):
        return self.zbetaj / self.scalex


    def _create_model(self):
        zX, y = self._prep_data()
        with pm.Model() as model:
            zbetaj = pm.Normal('zbetaj', mu=0, sd=2, shape=zX.shape[1])

            p = pm.invlogit(zbeta0 + pm.math.dot(zX, zbetaj))
            likelihood = pm.Bernoulli('likelihood', p, observed=y, total_size=y.shape[0])  # noqa
            #  pm.model_to_graphviz(model)
        return model

    def plot_posterior(self, save_path=None, axes_size=4, shape=None, credible_interval=0.94,
                       color_bad=True):
        return self._plot_posterior(self.betaj, save_path, axes_size, shape,
                                    credible_interval, color_bad)

    def plot_zposterior(self, save_path=None, axes_size=4, shape=None, credible_interval=0.94,
                        color_bad=True):
        return self._plot_posterior(self.zbetaj, save_path, axes_size, shape,
                                    credible_interval, color_bad)

