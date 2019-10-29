# core imports
#  from typing import Mapping, Collection
import concurrent.futures
import collections
import typing
from collections import abc

# 3rd party imports
import pandas as pd
import numpy as np
import seaborn as sns
import tqdm
from peputils.proteome import fasta_to_protein_hash
from sequtils import SequenceRange

# local
from .protein import ProteinFeatureExtractor
from .split import XFold


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
            return fasta_to_protein_hash(proteome)
        elif isinstance(proteome, abc.Mapping):
            return proteome
        error = "argument proteome is not of type 'str' or 'mapping' but {}"
        raise ValueError(error.format(type(proteome)))


@pd.api.extensions.register_dataframe_accessor("ppv")
class PandasPPV:
    """
    This object manipulates a UPF dataframe and adds the features nessesary for
    prediction peptide variants (PPV)
    such as extracting feature from the upf data frame
    """

    def __init__(self, df):
        _validate(df)
        self.df = df
        self.n_samples = self.df.shape[1]

    def create_feature_df(self,
                          protein_sequences: typing.Union[str, typing.Dict[str, set]],
                          delta_imp: int = 4,
                          peptides: str = 'valid',
                          known: typing.Union[None, str, typing.Dict[str, set]] = None,
                          normalize: bool = False,
                          disable_progress_bar=False,
                          n_cpus=4):
        known = ArgumentConverter.resolve_known_peptides(known)
        df = self.df
        if normalize:
            df = df.peptidomics.normalize()
        median = np.nanmedian(df.values.flatten())

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
                pfe, df_potein = self._get_protein_features(df_protein, sequence, median,
                                                            delta_imp, peptides, known_peptides)
                features.append(df_protein)
        else:
            exe = concurrent.futures.ProcessPoolExecutor(n_cpus)
            for protein_id, df_protein in self.df.groupby(level='protein_id'):
                known_peptides = known.get(protein_id, set())
                if len(known_peptides) == 0 and peptides == 'fair':
                    progress_bar.update(1)
                    continue
                sequence = protein_sequences[protein_id]
                future = exe.submit(self._get_protein_features, df_protein, sequence, median,
                                    delta_imp, peptides, known_peptides)
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

    def _get_protein_features(self, df_protein, protein_sequence, median, delta_imp, peptides,
                              known_peptides):
        pfe = ProteinFeatureExtractor(df_protein, protein_sequence, median, known_peptides)
        return pfe, pfe.create_feature_df(delta_imp, peptides)


@pd.api.extensions.register_dataframe_accessor("ppv_features")
class PandasPPVFeatures:
    """
    This object manipulates the ppv feature object
    """

    def __init__(self, df):  # , protein_features=None):
        _validate(df)
        self.df = df

    def plot(self, path, show=False):
        sns.set(style="ticks", color_codes=True)
        # TODO: make known bool the correct place!!!!
        positives = self.df[self.df["Target", "known"]]
        negatives = self.df[~self.df["Target", "known"]].sample(positives.shape[0] * 10)
        data = pd.concat((positives, negatives))

        g = sns.PairGrid(data, hue=("Target", "known"), hue_kws={"cmap": ["Greens", "Reds"]})
        #  g = g.map_diag(sns.kdeplot, lw=3)
        #  g = g.map_offdiag(sns.kdeplot, lw=1)
        g = g.map_diag(sns.kdeplot)
        g = g.map_offdiag(sns.kdeplot)
        g.savefig(path)
        #  peptide_features.ppv_features.generate_xfolds()
        #  print(peptide_features)

    def train(self):
        #  y = self.df["Target", "known"]
        #  features = self.df
        raise NotImplementedError("TODO!!")

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
