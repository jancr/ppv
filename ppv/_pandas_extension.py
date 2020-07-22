# core imports
#  from typing import Mapping, Collection
import concurrent.futures
import collections
import typing
from collections import abc
#  import pickle
import math
import warnings

# 3rd party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from sequtils import SequenceRange

# local
from .protein import ProteinFeatureExtractor
from .split import XFold
from .model import PPVModel

#  mpl.use('agg')
#  Grå: R230, G231, B231
#  Blå: R81, G96, B171


def _validate(df):
    if not isinstance(df, pd.DataFrame):
        raise AttributeError("df is not a pandas DataFrame!")


Known = collections.namedtuple("Known", ("protein_id", "start", "stop", "seq", "mod_seq", "type",
                               "full_name", "short_name"))


class ArgumentConverter:
    @classmethod
    def get_known_peptides(cls, known_file: str) -> typing.Dict[str, set]:
        known_peptides = collections.defaultdict(set)
        with open(known_file) as known_file:
            known_file.readline()  # skip header
            for line in known_file:
                known = Known(*line.rstrip('\r\n').split('\t'))
                if known.type in ('peptide', 'propeptide'):
                    peptide = SequenceRange(int(known.start), int(known.stop), seq=known.seq)
                    known_peptides[known.protein_id].add(peptide)
        return dict(known_peptides)

    @classmethod
    def resolve_known_peptides(cls, known_peptides: [None, str, typing.Dict[str, set]]
                               ) -> typing.Dict[str, set]:
        if known_peptides is None:
            return {}
        elif isinstance(known_peptides, abc.Collection):
            if isinstance(known_peptides, str):
                return cls.get_known_peptides(known_peptides)
            return known_peptides
        error = "known_peptides must be of type None, str or Mapping, not {}"
        raise ValueError(error.format(type(known_peptides)))

    @classmethod
    def get_proteome(cls, proteome: typing.Union[str, typing.Dict[str, str]]
                     ) -> typing.Dict[str, str]:
        if isinstance(proteome, str):
            return cls._fasta_to_protein_hash(proteome)
        elif isinstance(proteome, abc.Mapping):
            return proteome
        error = "argument proteome is not of type 'str' or 'mapping' but {}"
        raise ValueError(error.format(type(proteome)))

    def _fasta_to_protein_hash(cls, fasta_file):
        proteins = {}
        with open(fasta_file) as fasta_file:
            acc = None
            for line in fasta_file:
                if line.startswith(">"):
                    acc = line.rstrip("\r\n>").lstrip('>').split(' ')[0]
                    proteins[acc] = ""
                else:
                    proteins[acc] += line.rstrip()
            return proteins


@pd.api.extensions.register_dataframe_accessor("ppv_feature_extractor")
class PandasPPV:
    """
    This object manipulates a UPF dataframe and adds the features nessesary for
    prediction peptide variants (PPV)
    """

    def __init__(self, df):
        _validate(df)
        self.df = df
        self.n_samples = self.df.shape[1]

    def create_feature_df(self,
                          protein_sequences: typing.Union[str, typing.Dict[str, set]],
                          #  delta_imp: int = 4,
                          peptides: str = 'valid',
                          known: typing.Union[None, str, typing.Dict[str, set]] = None,
                          normalize: bool = False,
                          disable_progress_bar=False,
                          n_cpus=4):
        known = ArgumentConverter.resolve_known_peptides(known)
        df = self.df
        if normalize:
            df = df.peptidomics.normalize()
        #  median = np.nanmedian(df.values.flatten())

        features = []
        futures = []
        progress_bar = tqdm.tqdm("Creating Features", total=self.n_proteins,
                                 disable=disable_progress_bar)
        if n_cpus == 1:
            for protein_id, df_protein in self.df.groupby(level='protein_id'):
                known_peptides = known.get(protein_id, set())
                progress_bar.update(1)
                if len(known_peptides) == 0 and peptides == 'fair':
                    continue
                sequence = protein_sequences[protein_id]
                pfe, df_potein = self._get_protein_features(df_protein, sequence,
                                                            peptides, known_peptides)
                features.append(df_protein)
        else:
            exe = concurrent.futures.ProcessPoolExecutor(n_cpus)
            for protein_id, df_protein in self.df.groupby(level='protein_id'):
                known_peptides = known.get(protein_id, set())
                if len(known_peptides) == 0 and peptides == 'fair':
                    progress_bar.update(1)
                    continue
                sequence = protein_sequences[protein_id]
                future = exe.submit(self._get_protein_features, df_protein, sequence,
                                    peptides, known_peptides)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                pfe, df_protein = future.result()
                features.append(df_protein)
                progress_bar.update(1)

        df_features = pd.concat(features)
        return df_features

    @property
    def n_proteins(self):
        return len(self.df.index.get_level_values('protein_id').unique())

    def _get_protein_features(self, df_protein, protein_sequence, peptides,
                              known_peptides):
        pfe = ProteinFeatureExtractor(df_protein, protein_sequence, known_peptides)
        return pfe, pfe.create_feature_df(peptides)


@pd.api.extensions.register_series_accessor("ppv")
class PandasSeriesPPVFeatures:
    def __init__(self, series):
        self.series = series

    def diversify_score(self, dampening=2.0):
        """
        This method modifies a series of scores (usally ppv prediction) and tries to make a
        tradeoff between high predictions and high similarity to previous
        predictions, the algorithm to achive this is very simple:
        1) take the highest score
        2) downvote all who share amino acids with it by a factor of 'dampening'
        3) goto 1
        """
        # create overlap[campaign][proteinid] = {SequenceRange(..)..}
        # so we can quickley find overlapping peptides
        peptides = collections.defaultdict(lambda: collections.defaultdict(set))
        for campaign_id, entry_id, score in self.series.peputils.iteritems(keep_campaign_id=True):
            sequence_range = SequenceRange(entry_id.start, entry_id.stop)
            peptides[campaign_id][entry_id.protein_id].add(sequence_range)

        # the algorithm
        scores = self.series.copy()
        new_score = pd.Series(index=self.series.index)
        while scores.shape[0] != 0:
            # add best score to new_score
            best_id = scores.idxmax()
            new_score[best_id] = scores[best_id]
            # penaltize overlapping peptides
            overlapping_ids = self._get_overlaping_ids(*best_id, peptides)
            scores[overlapping_ids] /= dampening
            del scores[best_id]
        return new_score

    def _get_overlaping_ids(campaign_id, protein_id, start, stop, overlap):
        peptides = overlap[campaign_id][protein_id]
        best_peptide = SequenceRange(start, stop)
        return [(campaign_id, protein_id, peptide.start.pos, peptide.stop.pos)
                for peptide in peptides
                if peptide.contains(best_peptide, part=any)]


@pd.api.extensions.register_dataframe_accessor("ppv")
class PandasDataFramePPVFeatures:
    """
    This object manipulates a DataFrame containing ppv feature object
    """

    def __init__(self, df):
        _validate(df)
        self.df = df

    @property
    def predictors(self):
        if "Annotations" in self.df.columns:
            return self.df.drop("Annotations", axis=1)
        return self.df.copy()

    @property
    def positives(self):
        return self.df[self.target.astype(bool)]

    @property
    def negatives(self):
        return self.df[~self.target.astype(bool)]

    @property
    def target(self):
        return self.df["Annotations", "Known"]

    @property
    def observed(self):
        df = self.df[self.df["MS Bool", "observed"]].copy()
        del df["MS Bool", "observed"]
        return df

    def get_prior(self):
        return self.target.astype(bool).sum() / self.target.shape[0]

    @classmethod
    def _seq_to_modseq(cls, row):
        mod_seq = "_{{n}}{}_{{c}}".format(row["Annotations", "Sequence"])
        n_term = c_term = ""
        if row["MS Frequency", "acetylation"] > 0.05:
            n_term = "(ac)"
        if row["MS Frequency", "amidation"] > 0.05:
            c_term = "(am)"
        return mod_seq.format(n=n_term, c=c_term)

    def to_variants(self):
        df = self.df.copy()
        df["mod_seq"] = df.apply(self._seq_to_modseq, axis=1)
        df["origin"] = "collapsed"
        return df.set_index("mod_seq", append=True).set_index("origin", append=True)

    #  @classmethod
    #  def load_model(cls, path):
    #      clf, scaler = pickle.loads(open(path, 'rb'))

    #  @classmethod
    #  def save_model(clf, scaler, path="pickle/model.pickle"):
    #      pickle.dump((clf, scaler), open(path, "wb"))

    def _transform(self):
        """
        Transforms Features so they are more predictive
        This is achived by changing
            changing charge to net charge (difference from zero)
            changing pI to "distance from neutral"
        """
        if "Chemical" not in self.df:
            warnings.warn("Chemical is not in the index...")
            return self.df

        df_transformed = self.df.rename(columns={"Charge": "Net Charge",
                                                 "ChargeDensity": "Net ChargeDensity",
                                                 "pI": "abs(pI - 7)"})

        chem = self.df["Chemical"]
        if all(chem.columns == df_transformed["Chemical"].columns):
            warnings.warn("Chemical seems to be already transformed")
            return self.df

        if "Charge" in chem:
            df_transformed["Chemical", "Net Charge"] = chem["Charge"].abs()
        if "ChargeDensity" in chem:
            df_transformed["Chemical", "Net ChargeDensity"] = chem["ChargeDensity"].abs()
        if "pI" in chem:
            df_transformed["Chemical", "abs(pI - 7)"] = (chem["pI"] - 7).abs()
        return df_transformed

        #  features_fixed = self.df.copy()
        #  #  for bad_feature in ("Charge", "ChargeDensity", "pI"):
        #  #      if bad_feature in features_fixed:
        #  #          del features_fixed["Chemical", bad_feature]
        #  if ("Chemical", "Charge") in self.df:
        #      features_fixed["Chemical", "Net Charge"] = self.df["Chemical", "Charge"].abs()
        #      del features_fixed["Chemical", "Charge"]
        #  if ("Chemical", "ChargeDensity") in self.df:
        #      _cd = self.df["Chemical", "ChargeDensity"].abs()
        #      features_fixed["Chemical", "Net ChargeDensity"] = _cd
        #      del features_fixed["Chemical", "ChargeDensity"]
        #  if ("Chemical", "pI") in self.df:
        #      features_fixed["Chemical", "abs(pI - 7)"] = (self.df["Chemical", "pI"] - 7).abs()
        #      del features_fixed["Chemical", "pI"]
        #  #  features_fixed.pop("Annotations")
        #  return features_fixed

    #  def _get_scaler(self):
    #      return StandardScaler().fit(self.predictors)
    #
    #  def _scale_predictors(self, scaler=None):
    #      if scaler is None:
    #          scaler = self.get_scaler()
    #      predictors = self.predictors
    #      predictors_scaled = pd.DataFrame(scaler.transform(predictors), index=predictors.index,
    #                                       columns=predictors.columns)
    #      for name, series in self.df["Annotations"].iteritems():
    #          predictors_scaled["Annotations", name] = series
    #      return predictors_scaled

    #  def _train(self, **kwargs):
    #      model = LogisticRegression(random_state=0, max_iter=5000, **kwargs)
    #      return model.fit(self.predictors, self.target.astype(int))

    #  def predict(self, model):
    #      predictors = self._transform().ppv.predictors
    #
    #      if isinstance(model, str):
    #          model = joblib.load(model)
    #      predictions = model.predict_proba(predictors)[:, 1]
    #      return pd.Series(predictions, index=self.df.index, name=("Annotations", "PPV"))

    #  def create_model(self, path=None, *, random_state=0, max_iter=5000, **kwargs):
    #      features_transformed = self._transform()
    #
    #      scaler = features_transformed.ppv._get_scaler()
    #      logistic_model = LogisticRegression(random_state=random_state,
    #                                          max_iter=max_iter, **kwargs)
    #      pipline = make_pipeline(scaler, logistic_model)
    #
    #      model = pipline.fit(features_transformed.ppv.predictors, self.target.astype(int))
    #      if path:
    #          joblib.dump(model, path)
    #      return model

    def transform_to_null_features(self):
        # drop all features and select only observed
        df = self.subset({})[self.df["MS Bool", "observed"]]
        df["Null", "Log10(Intensity)"] = np.log10(df["Annotations", "Intensity"])
        return df

    def _class_balance(self, target, class_balance):
        known_selector = self.df[target]
        positives = self.df[known_selector]
        negatives = self.df[~known_selector]
        if class_balance is not None:
            n_negatives = self.df.shape[0] - positives.shape[0]
            negatives = negatives.sample(frac=class_balance * positives.shape[0] / n_negatives)
        return pd.concat((positives, negatives))

    def plot_pair_grid(self, path=None, class_balance=None, target=("Annotations", "Known")):
        sns.set(style="ticks", color_codes=True)
        data = self._class_balance(target, class_balance)
        g = sns.PairGrid(data.sample(frac=1), hue=target, hue_kws={"cmap": ["Greens", "Reds"]})
        g = g.map_diag(plt.hist)
        g = g.map_offdiag(plt.scatter)
        if isinstance(path, str):
            g.savefig(path)
        return g

    def subset(self, feature_types, keep_annotations=True):
        if isinstance(feature_types, str):
            feature_types = {feature_types}
        feature_types = set(feature_types)

        if keep_annotations:
            feature_types.add("Annotations")
        all_feature_types = set(self.df.columns.get_level_values(0))
        feature_types_to_drop = all_feature_types - set(feature_types)
        return self.drop(feature_types_to_drop)

    def drop(self, feature_types):
        if isinstance(feature_types, str):
            feature_types = {feature_types}
        df = self.df.copy()
        for feature_type in feature_types:
            df.drop(feature_type, inplace=True, axis=1)
        return df

    def _predictor_canvas(self, axes_size=4, extra=0):
        features = self.predictors.shape[-1] + extra
        n_col = math.ceil(features ** 0.5)
        n_row = math.ceil(features / n_col)
        fig = plt.figure(figsize=(n_col * axes_size, n_row * axes_size))
        axes = fig.subplots(n_row, n_col)
        return fig, axes

    def create_histograms(self, axes_size=2, shape=None):
        df = self.df.copy()
        #  if features is not None:
        #      df = df[list(features) + "Annotations"]
        known = df.ppv.positives.ppv.predictors
        unknown = df.ppv.negatives.ppv.predictors

        #  fig, axes = self._predictor_canvas(axes_size)
        fig, axes = PPVModel._predictor_canvas(self.predictors, axes_size, shape=shape)
        for ax, feature in zip(axes.flatten(), df.ppv.predictors.columns):
            bins = np.linspace(df[feature].min(), df[feature].max(), 15)
            ax.hist(known[feature], bins, density=True, alpha=0.5, color='g')
            ax.hist(unknown[feature], bins, density=True, alpha=0.5, color='r')
            ax.set_title(' '.join(feature))
        return fig

    #  def plot_posterior(self, trace, meanx=None, scalex=None, save_path=None, axes_size=4):
    #      if meanx is not None and scalex is not None:
    #          beta0 = trace['zbeta0'] - np.sum(trace['zbetaj'] * meanx / scalex, axis=1)
    #          betaj = (trace['zbetaj'] / scalex)
    #      else:
    #          beta0 = trace['zbeta0'] - np.sum(trace['zbetaj'], axis=1)
    #          betaj = (trace['zbetaj'])
    #
    #      fig, axes = self._predictor_canvas(axes_size, 1)
    #      pm.plot_posterior(beta0, point_estimate='mode', ax=axes[0, 0], color=color)
    #      axes[0, 0].set_xlabel(r'$\beta_0$ (Intercept)', fontdict=f_dict)
    #      axes[0, 0].set_title('', fontdict=f_dict)
    #      columns = self.predictors.columns
    #      for i, (ax, feature) in enumerate(zip(axes.flatten()[1:], columns)):
    #          pm.plot_posterior(betaj[:, i], point_estimate='mode', ax=ax, color=color)
    #          ax.set_title('', fontdict=f_dict)
    #          ax.set_xlabel(r'$\beta_{{{}}}$ ({})'.format(i + 1, ' '.join(feature)),
    #                        fontdict=f_dict)
    #      if save_path is not None:
    #          fig.savefig(save_path)
    #      return fig

    def plot_hist(self, path=None, class_balance=None, target=("Annotations", "Known")):
        data = self._class_balance(target, class_balance)
        g = data.hist()
        if isinstance(path, str):
            g.savefig(path)
        return g

    def split(self, xfolds=5):
        raise NotImplementedError("TODO!!")

    def generate_xfolds(self, n_folds: int = 5, validation=None):
        # only count known peptides who are findable in the dataset...
        # thus a known peptide who does not share a start and end with a upf_peptid does not count
        # just like a "negative" that does not have a upf_start and end are not considered either
        #  n_known_peptides = collections.defaultdict(int)

        import colored_traceback.auto; import ipdb; ipdb.set_trace()  # noqa
        n_known_peptides = {}
        print(self.df["known"].sum())
        # self.findable = [{} for _ in range(self.n_upf_files)]
        #  for protein_id, known_peptides in self.known_peptides.items():
        #      for i, peptide_scorers in enumerate(self.peptide_scorers):
        #          if protein_id in peptide_scorers:
        #              ps = peptide_scorers[protein_id]
        #              if len(ps.findable) != 0 and len(ps.findable) != len(ps.valid_peptides):
        #                  n_known_peptides[protein_id] += len(ps.findable)

        xfold = XFold(n_folds, dict(n_known_peptides), validation)
        return xfold

    def _drop_weak_features(self):
        pi = ["pI", "abs(pI - 7)"]  # not credible when including MS features
        charge_density = ["ChargeDensity", "Net ChargeDensity"]
        base_weak_chemical = \
            ["AliphaticInd", "BomanInd", "eisenberg", "HydrophRatio"] + charge_density

        df = self.df.copy()
        feature_categories = set(list(zip(*list(self.predictors.columns)))[0])
        if feature_categories == {"Chemical"}:
            # keep the strong confitional on chemical features
            weak_features = {"Chemical": base_weak_chemical}
        else:
            # keep the strong conditional on full feature set
            weak_features = {
                "MS Intensity": ("penalty_start", "penalty_stop"),
                "MS Count": ("start", "stop"),
                "MS Frequency": ("cluster_coverage", "ladder"),
                "Chemical": base_weak_chemical + pi
            }
        for feature_type, features in weak_features.items():
            for feature in features:
                if (feature_type, feature) in df:
                    del df[feature_type, feature]
        return df

    def transform_to_chem_features(self):
        # is this the correct order, and does it matter?
        chem_labels = [("Chemical", "Net Charge"), ("Chemical", "abs(pI - 7)"),
                        ("Chemical", "InstabilityInd"), ("Chemical", "Aromaticity")]
        annotation_labels = [("Annotations", c) for c in self.df["Annotations"].columns]
        return self.df.filter(items=chem_labels + annotation_labels)

    def transform_to_core_features(self):
        core_labels = [("MS Intensity", "start"), ("MS Intensity", "stop"),
                       ("MS Bool", "first"), ("MS Bool", "last")]
        annotation_labels = [("Annotations", c) for c in self.df["Annotations"].columns]
        return self.df.filter(items=core_labels + annotation_labels)

    #  def zscore_predictors(data):
    #      # remember to transform first!
    #      X = data.predictors
    #      meanx = X.mean().values
    #      scalex = X.std().values
    #      zX = ((X - meanx) / scalex).values
    #      return zX

    #  def rescale_parameters(self,

    #  def predict(self, intercept, parameters, offset):
    #      df = self.df
