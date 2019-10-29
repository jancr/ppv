# core imports
import copy
from typing import Collection
import math
import itertools
import collections

# 3rd party imports
import numpy as np
import pandas as pd
#  from modlamp.sequences import Helices
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

# jcr imports
from sequtils import SequenceRange


FeatureTuple = collections.namedtuple("FeatureTuple", ("type", "features"))


class ProteinFeatureExtractor:
    ms_intensity_features = FeatureTuple("MS Intensity",
                                         ("d_start", "d_stop", "pd_start", "pd_stop"))
    ms_feequency_features = FeatureTuple("MS Frequency",
                                         ("f_ac", "f_am", "f_start", "f_stop", "f_obs"))
    chemical_features = FeatureTuple(
        "Chemical",
        ("Charge", "ChargeDensity", "pI", "InstabilityInd", "Aromaticity",
         "AliphaticInd", "BomanInd", "HydrophRatio", "eisenberg"))
    """
    This function holds all the matrices needed to score all peptides from one
    protein, "predict" is then called repeatetly to score each peptide
    """

    def __init__(self, df_protein: pd.DataFrame,
                 protein_sequence: str,
                 dataset_median: float,
                 #  n_samples: int,
                 #  length: Optional[int] = None,
                 #  protein_id: str,
                 known_peptides: Collection = frozenset()):
        if not isinstance(protein_sequence, str):
            raise AttributeError("protein_sequence must of type 'str'")
        self.length = len(protein_sequence)
        self.protein_sequence = protein_sequence
        self.log10_median = math.log10(dataset_median)

        #  if not isinstance(n_samples, int):
        #      raise ValueError("n_samples must be of type int")
        #  self.n_samples = n_samples
        #  self.protein_id = protein_id
        self.campaign_id, self.protein_id = df_protein.index[0][:2]
        self.n_samples = df_protein.shape[1]
        self.known_peptides = known_peptides
        self.upf_entries = {SequenceRange(pep_var_id.start, pep_var_id.stop): data.sum()
                            for pep_var_id, data in df_protein.peptidomics.iterrows()
                            if data.dropna().shape[0] != 0}
        self.df = df_protein

        self.h, self.h_start, self.h_stop, self.h_ac, self.h_am, self.h_bond = \
            self.make_histograms(self.df, self.length, self.n_samples)
        self.h_sample = self.make_sample_frequency_histogram(self.df)

        self.no_aa = self.h == 0
        self.h_cluster = self.get_cluster_histogram(self.no_aa)

        self.valid_starts, self.valid_stops = self.get_valid_positions(self.upf_entries,
                                                                       self.h_cluster)
        self.valid_peptides = self.get_valid_peptides(self.valid_starts, self.valid_stops)
        self.findable = set(self.known_peptides) & self.valid_peptides.keys()
        self.fair_peptides = self.get_fair_peptides(self.valid_peptides, self.findable)

        # calc % acetylations and methylations
        self.h_start_freq = self.normalize_histogram(self.h_start, self.h)
        self.h_stop_freq = self.normalize_histogram(self.h_stop, self.h)
        self.ac_freq = self.normalize_histogram(self.h_ac, self.h_start)
        self.am_freq = self.normalize_histogram(self.h_am, self.h_stop)

    def create_feature_df(self, delta_imp: int, peptides: str = 'valid'):
        if delta_imp < 0:
            raise ValueError("You cannot downshift by a negative number!")
        ########################################
        # derived
        ########################################
        # depends on w_imp_val
        imp_val = max(self.log10_median - delta_imp, 0)
        #  h10 = self.fake_log10(self.h, imp_val)
        self.h_overlap = np.zeros(self.length)

        #  h10_padded = np.ones(self.length + 2) * imp_val
        #  h10_padded[1:-1] = h10

        # depends on self.h10, which depends on imp_val
        self.start_scores, self.stop_scores = self.get_scores(self.h, imp_val)

        # get peptides and stabalize looping order, so we can use df.iloc
        peptides = list(sorted(self.get_peptides_by_type(peptides)))

        # pre allocate df
        #  annotations = ["known"]
        make_features = lambda t: zip(itertools.repeat(t.type), t.features)
        feature_list = (self.ms_intensity_features, self.ms_feequency_features,
                        self.chemical_features)
        features = list(itertools.chain(*map(make_features, feature_list)))
        features += [["Target", "known"]]

        _names = ('campaign_id', 'protein_id', 'start', 'stop')
        _tuples = [(self.campaign_id, self.protein_id, *p.pos) for p in peptides]
        index = pd.MultiIndex.from_tuples(_tuples, names=_names)
        columns = pd.MultiIndex.from_tuples(features)
        types = {c: float for c in columns}
        types['Target', 'known'] = bool

        df = pd.DataFrame(np.zeros((len(peptides), len(features))) * np.nan,
                          index=index, columns=columns).T
        for peptide in peptides:
            _index = (self.campaign_id, self.protein_id, peptide.start.pos, peptide.stop.pos)
            peptide_series = self._add_features_to_peptide_series(peptide, df.index)
            df[_index] = peptide_series
        return df.T.astype(types)

    # helper methods
    def _add_features_to_peptide_series(self, peptide, index):
        # primary intensity weights d = delta, pd = penalty delta
        # TODO only d_start and d_stop depends on impval, pd_start and pd_stop does not because
        # they are always between a d_start and d_stop, and should thus be above imp_val!
        # therefore we can write out d_start as and d_stop as:
        #   [before_start, after_start], [befrore_stop, after_stop]
        # thus if we have
        #       raw data     = [0, 0, 5, 5, 7, 7, 5, 5, 0, 0]
        # then for the peptide        3--------------8
        #       before_start, after_start = [ 0, 5 ]
        # but for the peptide               5--6
        #       before_start, after_start = [ 5, 7 ]
        # by making a none linear model we could formulate the w_start parameter as follows:
        # w_start * (after_start - max(before_start, imp_val))
        # which is consistent with how we currently do the grid search (imp_val=4):
        #       d_start = 5 - max(0, 4) = 1
        #       d_start = 7 - max(5, 4) = 2
        series = pd.Series(np.zeros(len(index)) * np.nan, index=index)
        ms_int = self.ms_intensity_features.type
        series[ms_int, 'd_start'] = self.start_scores[peptide.start.index]
        series[ms_int, 'd_stop'] = self.stop_scores[peptide.stop.index]

        if 4 < len(peptide):
            penalty = SequenceRange(peptide.start + 1, peptide.stop - 1, validate=False)
            series[ms_int, 'pd_start'] = self.start_scores[penalty.slice].sum()
            series[ms_int, 'pd_stop'] = self.stop_scores[penalty.slice].sum()
        else:
            series[ms_int, 'pd_start'] = series[ms_int, 'pd_stop'] = 0

        # ptm weights
        # TODO: should it get extra penalties if there are PTM's between start and end?
        ms_freq = self.ms_feequency_features.type
        series[ms_freq, 'f_ac'] = self.ac_freq[peptide.start.index]
        series[ms_freq, 'f_am'] = self.am_freq[peptide.stop.index]

        series[ms_freq, 'f_start'] = self.h_start_freq[peptide.start.index]
        series[ms_freq, 'f_stop'] = self.h_stop_freq[peptide.stop.index]
        series[ms_freq, 'f_obs'] = self._calc_f_obs(peptide)

        # TODO bonds!!!!

        sequence = self.protein_sequence[peptide.slice]
        peptide_features = GlobalDescriptor(sequence)

        is_amidated = series[ms_freq, 'f_am'] > 0.05
        peptide_features.calculate_all(amide=is_amidated)

        chem = self.chemical_features.type
        for i, name in enumerate(peptide_features.featurenames):
            if name in self.chemical_features.features:
                series[chem, name] = peptide_features.descriptor[0, i]

            eisenberg = PeptideDescriptor(sequence, 'eisenberg')
            eisenberg.calculate_moment()
            series[chem, 'eisenberg'] = eisenberg.descriptor.flatten()[0]
        series["Target", "known"] = peptide in self.known_peptides

        return series

    def _calc_f_obs(self, peptide):
        if peptide not in self.upf_entries:
            return 0
        obs_area = self.upf_entries[peptide] * peptide.length / self.n_samples
        h_area = self.h[peptide.slice].sum()
        return obs_area / h_area

    def get_peptides_by_type(self, type_: str):
        peptide_types = {'valid': self.valid_peptides, 'fair': self.fair_peptides}
        if type_ not in peptide_types:
            raise ValueError(f"{type_} not in {peptide_types.keys()}")
        return peptide_types[type_]

    ########################################
    # Histogram manipulation
    ########################################
    @classmethod
    def normalize_histogram(cls, histogram, bg_histogram):
        h = np.zeros(len(bg_histogram))
        not_zero = histogram != 0
        h[not_zero] = histogram[not_zero] / bg_histogram[not_zero]
        return h

    def fake_log10(self, data, imp_val=1):
        "normalizes data to imp_val, default is 1 as this will make all 0 stay 0"
        data = data.copy()
        # if a peptide > imputation score, it would get a 'negative start score!!!'
        data[data <= 10 ** imp_val] = 10 ** imp_val
        return np.log10(data)

    def get_scores(self, histogram, imp_val):
        with np.errstate(divide='raise'):
            h10 = self.fake_log10(histogram, imp_val)
            #  self.h_overlap = np.zeros(self.length)

            h10_padded = np.ones(self.length + 2) * imp_val
            h10_padded[1:-1] = h10

            start_scores = h10 - h10_padded[:-2]
            start_scores[start_scores < 0] = 0

            stop_scores = h10 - h10_padded[2:]
            stop_scores[stop_scores < 0] = 0
            return start_scores, stop_scores

    ########################################
    # Parameter maipulation
    ########################################
    def predict(self, p):
        raise NotImplementedError("this does not work because we are now useing DataFrames")

    def iter_fair(self):
        for p in self.fair_peptides:
            yield p, self.predict(p)

    def get_all_predictions(self, max_overlap=None):
        self.h_overlap[:] = 0
        predictions = {}
        for p, score in self:
            predictions[p] = score
        if max_overlap is not None:
            self.reduce_overlap(predictions, max_overlap)
        return predictions

    def iter_observed(self):
        for upf_entry in self.upf_entries:
            yield upf_entry, self.predict(upf_entry)

    def get_obsered_predictions(self):
        return dict(self.iter_observed())

    def get_fair_predictions(self, max_overlap=None):
        self.h_overlap[:] = 0
        predictions = {p: score for p, score in self.iter_fair()}
        if max_overlap is not None:
            self.reduce_overlap(predictions, max_overlap)
        return predictions

    def reduce_overlap(self, predictions, max_overlap):
        """
        this function removes violations scoring peptides from predictions
        untill it's overlap is lower than max_overlap
        """
        h_overlap = self.h_overlap.copy()
        violations = copy.deepcopy(predictions)
        good = []
        violations = self._get_violations(violations, good, h_overlap, max_overlap)
        while violations:
            (score, peptide) = min((score, peptide) for (peptide, score) in violations.items())
            del violations[peptide]
            del predictions[peptide]
            h_overlap[peptide.slice] -= 1
            violations = self._get_violations(violations, good, h_overlap, max_overlap)

    @classmethod
    def _get_violations(cls, violations, good, h_overlap, max_overlap):
        still_violations = {}
        for p, score in violations.items():
            if h_overlap[p.slice].max() <= max_overlap:
                good.append(p)
            else:
                still_violations[p] = score
        return still_violations

    @classmethod
    def make_histograms(cls, df, length, n_samples):
        histogram = np.zeros(length)
        histogram_start = np.zeros(length)
        histogram_stop = np.zeros(length)
        histogram_ac = np.zeros(length)
        histogram_am = np.zeros(length)
        histogram_bonds = np.zeros(length - 1)

        #  for upf_entry in upf_entries:
        for pep_var_id, peptide_series in df.peptidomics.iterrows():
            p = SequenceRange(pep_var_id.start, pep_var_id.stop)
            intensity = peptide_series.sum() / n_samples
            if pep_var_id.mod_seq.startswith('_(ac)'):
                histogram_ac[p.start.index] += intensity
            if pep_var_id.mod_seq.endswith('_(am)'):
                histogram_am[p.stop.index] += intensity
            histogram_start[p.start.index] += intensity
            histogram_stop[p.stop.index] += intensity
            histogram[p.slice] += intensity
            histogram_bonds[p.slice.start:p.slice.stop - 1] += intensity

        return (histogram, histogram_start, histogram_stop, histogram_ac, histogram_am,
                histogram_bonds)

    #  @classmethod
    def make_sample_frequency_histogram(self, df):
        histogram_samples = pd.DataFrame(np.zeros((self.length, df.shape[1])),
                                         columns=df.columns)
        for pep_var_id, peptide_series in df.peptidomics.iterrows():
            p = SequenceRange(pep_var_id.start, pep_var_id.stop)
            for group, intensity in peptide_series.dropna().iteritems():
                histogram_samples[group][p.slice] = 1
        if not (0 <= histogram_samples.shape[1] <= self.n_samples):
            raise ValueError("max_samples, higher than the accual number of samples!!!")
        return histogram_samples.sum(axis=1).values / self.n_samples

    ########################################
    # valid/fair
    ########################################
    @classmethod
    def get_cluster_histogram(cls, no_aa):
        cluster = np.zeros(no_aa.shape)
        cluster_count = 0
        in_cluster = False
        for i, aa in enumerate(~no_aa):
            if aa:
                if not in_cluster:
                    in_cluster = True
                    cluster_count += 1
                cluster[i] = cluster_count
            else:
                in_cluster = False
                cluster[i] = 0
        return cluster

    def get_fair_peptides(self, valid_peptides, findable):
        """
        A 'fair' peptide, is a peptide who resides in a 'block' where a known
        also resides, thus while most of them are "negatives", they are not all
        negatives and thus provides much better training examples than clusters
        where all peptides are negatives!
        """

        fair_clusters = {valid_peptides[peptide] for peptide in findable}
        return {peptide for peptide, cluster in valid_peptides.items()
                if cluster in fair_clusters}

        #  fair_peptides = set()
        #  for peptide in valid_peptides:
        #      for known_peptide in known_peptides:
        #          # if starts before
        #          if peptide.start.pos < known_peptide.start.pos:
        #              #  if no_aa[peptide.end_slice:known_peptide.start_slice].sum() == 0:
        #              if no_aa[peptide.slice.stop:known_peptide.slice.start].sum() == 0:
        #                  fair_peptides.add(peptide)
        #                  break
        #          # if ends after
        #          elif peptide.stop.pos > known_peptide.stop.pos:
        #              if no_aa[known_peptide.slice.stop:peptide.slice.start].sum() == 0:
        #                  #  fair_peptides.add(peptide.get_pos())
        #                  fair_peptides.add(peptide)
        #                  break
        #          # then it is alwaus in the same cluster!
        #          else:
        #              fair_peptides.add(peptide)
        #              break
        #  return fair_peptides

    @classmethod
    def get_valid_positions(cls, peptides, h_cluster):
        valid_starts = {}
        valid_stops = {}
        #  for upf_entry in upf_entries:
        #      p = SequenceRange(upf_entry.start, upf_entry.end)
        for p in peptides:
            valid_starts[p.start.pos] = h_cluster[p.start.index]
            valid_stops[p.stop.pos] = h_cluster[p.stop.index]
        return valid_starts, valid_stops

    @classmethod
    def get_valid_peptides(cls, valid_starts, valid_stops):
        valid_peptides = {}
        for v_start, c_start in valid_starts.items():
            for v_stop, c_stop in valid_stops.items():
                if v_start < v_stop and c_start == c_stop:
                    p = SequenceRange(v_start, v_stop)
                    valid_peptides[p] = c_start
        return valid_peptides

    ########################################
    # kelstrup algorithm
    ########################################
    def predict_lpvs(self, min_overlap=2, ptm_flag=True):
        """
        the LPV (Longest Peptide Variant) algorithm
        kelstrups algorithm can be simplified to the following steps
        for a sorted list, the next peptide will always have the same or a higher pep_begin
        the algorithm
        Phase 1)
             step 1) build a stack of sorted peptides, pop the first peptide off as a "lpv
             scaffold"
             step 2) pop off peptide, check if the peptide is within the lpv we are building
        if true redo step 2
        if false, calcuate the overlap
         if the overlap is above 2, then extend, and goto 2)
         otherwise use this peptide to start building the next lpv, goto step 2)
        Phase 2)
        for each build lpv, check if there are any start/ends have ptm's, if som do split the lpv
        accordingly
        """

        lpvs = self.build_lpv(min_overlap)
        if not ptm_flag:
            return [(lpv.start.pos, lpv.stop.pos) for lpv in lpvs]
        else:
            return self.split_ptm(lpvs)

    #  def __init__(self, features, min_overlap=2, ptm_flag=True):
    #      self.features = features
    #      self.min_overlap = min_overlap
    #      self.ptm_flag = ptm_flag

    # todo: make into iterator
    def _build_lpv(self, min_overlap):
        # phase 1)
        peptides = []
        for peptide in self.upf_entries:
            #  peptides.append([upf_entry.start, upf_entry.end])
            # TODO: use PeptideLocation instead of .start/stop.pos
            peptides.append([peptide.start.pos, peptide.stop.pos])

        # Phase 1)
        peptides.sort(reverse=True)
        predictions = []
        lpv_start, lpv_stop = peptides.pop()
        while len(peptides) != 0:
            pep_start, pep_stop = peptides.pop()

            # Step 1) you are inside the peptide - ignore/delete -
            if lpv_start <= pep_start <= lpv_stop and lpv_start <= pep_stop < lpv_stop:
                continue

            # Step 2) you are etending the peptide - extend -
            overlap = lpv_stop - pep_start + 1
            if overlap >= min_overlap:
                lpv_stop = pep_stop
                continue

            # no extension, no internal -> new lpv
            predictions.append(SequenceRange(lpv_start, lpv_stop))
            lpv_start = pep_start
            lpv_stop = pep_stop
        predictions.append(SequenceRange(lpv_start, lpv_stop))
        return predictions

    def split_ptm(self, predictions):
        # Phase 2)
        all_predictions = []
        for lpv in predictions:
            #  all_predictions.append((lpv.start.pos, lpv.stop.pos))
            ac = self.h_ac[lpv.slice]
            starts = {lpv.start.pos}
            if ac.sum() != 0:
                # this should be vectorized :D
                for pep_start, intensity in enumerate(ac, lpv.start.pos):
                    if intensity != 0:
                        starts.add(pep_start)

            am = self.h_am[lpv.slice]
            stops = {lpv.stop.pos}
            if am.sum() != 0:
                # this should be vectorized :D
                for pep_stop, intensity in zip(range(lpv.stop.pos, 0, -1), am[::-1]):
                    if intensity != 0:
                        stops.add(pep_stop)
            for s in starts:
                for e in stops:
                    all_predictions.append(SequenceRange(s, e))
        return all_predictions
