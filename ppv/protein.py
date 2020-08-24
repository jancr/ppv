# core imports
import copy
from typing import Collection
#  import math
import itertools
import collections
#  import typing

# 3rd party imports
import numpy as np
import pandas as pd
#  from modlamp.sequences import Helices
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

# jcr imports
from sequtils import SequenceRange


#  np.seterr(all='raise')


FeatureTuple = collections.namedtuple("FeatureTuple", ("type", "features"))
FeatureTupleDTyped = collections.namedtuple("FeatureTupleDTyped", ("type", "features", "dtypes"))
PeptideVariantId = collections.namedtuple('PeptideVariantId',
                                ('protein_id', 'start', 'stop', 'mod_seq', 'origin'))


class ProteinFeatureExtractor:
    """
    This function holds all the matrices needed to create MS and Chemical features for peptides
    """

    ms_intensity_features = FeatureTuple(
        "MS Intensity",
        ("start", "stop", "penalty_start", "penalty_stop"))
    ms_bool_features = FeatureTuple("MS Bool", ("first", "last", "observed"))
    ms_frequency_features = FeatureTuple(
        "MS Frequency",
        ("acetylation", "amidation", "start", "stop", "observed", "bond", "sample", "ladder",
         "protein_coverage", "cluster_coverage"))
    ms_count_features = FeatureTuple("MS Count", ("start", "stop"))
    chemical_features = FeatureTuple(
        "Chemical",
        ("Charge", "ChargeDensity", "pI", "InstabilityInd", "Aromaticity",
         "AliphaticInd", "BomanInd", "HydrophRatio", "eisenberg"))
    annotations = FeatureTupleDTyped(
        "Annotations",
        ("Known", "Cluster", "Intensity", "Sequence", "N Flanking", "C Flanking", "LPV"),
        ("category", int, float, str, str, str, bool))

    def __init__(self, df_protein: pd.DataFrame,
                 protein_sequence: str,
                 #  dataset_median: float,
                 known_peptides: Collection = frozenset()):
        if not isinstance(protein_sequence, str):
            raise AttributeError("protein_sequence must of type 'str'")
        self.length = len(protein_sequence)
        self.protein_sequence = protein_sequence
        #  self.log10_median = math.log10(dataset_median)

        self.campaign_id, self.protein_id = df_protein.index[0][:2]
        self.n_samples = df_protein.shape[1]
        self.known_peptides = known_peptides
        self.upf_entries = {SequenceRange(pep_var_id.start, pep_var_id.stop,
                                          full_sequence=self.protein_sequence): data.sum()
                            for pep_var_id, data in self.iterrows(df_protein)
                            if data.dropna().shape[0] != 0}
        self.df = df_protein

        (self.h, self.h_start, self.h_stop, self.h_ac, self.h_am, self.h_bond, self.h_first,
            self.h_last) = self.make_histograms(self.df, self.length, self.n_samples)
        self.h_sample = self.make_sample_frequency_histogram(self.df)

        self.no_aa = self.h == 0
        self.h_cluster = self.get_cluster_histogram(self.no_aa)
        self.clusters = self.get_clusters(self.h_cluster)
        self.protein_coverage = self.calc_coverage(self.no_aa)
        self.cluster_coverage = self.calcuate_cluster_coverage(self.clusters, self.h_cluster,
                                                               self.upf_entries)

        self.valid_starts, self.valid_stops = self.get_valid_positions(self.upf_entries,
                                                                       self.h_cluster)
        self.valid_peptides = self.get_valid_peptides(self.valid_starts, self.valid_stops,
                                                      self.protein_sequence)
        self.findable = set(self.known_peptides) & self.valid_peptides.keys()
        self.fair_peptides = self.get_fair_peptides(self.valid_peptides, self.findable)

        # counts
        self.start_counts, self.stop_counts = self.get_possition_counts(self.upf_entries)

        # calc % acetylations and methylations
        self.h_start_freq = self.normalize_histogram(self.h_start, self.h)
        self.h_stop_freq = self.normalize_histogram(self.h_stop, self.h)
        self.ac_freq = self.normalize_histogram(self.h_ac, self.h_start)
        self.am_freq = self.normalize_histogram(self.h_am, self.h_stop)

        # create ladders
        #  self.h_ladder_start = self.count_ladders(self.valid_starts, self.h_cluster,
        #                                           ladder_window=10)
        #  self.h_ladder_stop = self.count_ladders(self.valid_stops, self.h_cluster, self.clusters,
        #                                          ladder_window=10)
        self.h_ladder_start = self.count_ladders(self.start_counts, self.h_cluster, self.clusters,
                                                 ladder_window=10)
        self.h_ladder_stop = self.count_ladders(self.stop_counts, self.h_cluster, self.clusters,
                                                ladder_window=10)
        self.lpv_creator = LPVCreator(list(self.upf_entries.keys()), self.protein_sequence,
                                      self.h_ac, self.h_am)

    def create_feature_df(self, peptides: str = 'valid', lpv_column=True):
        ########################################
        # derived
        ########################################
        self.start_scores, self.stop_scores = self.get_terminal_histograms(
            self.h, self.h_start, self.h_stop)

        peptides = sorted((p, c) for (p, c) in self.get_peptides_by_type(peptides).items())

        # pre allocate df
        make_features = lambda t: zip(itertools.repeat(t.type), t.features)
        feature_list = (
            self.ms_intensity_features, self.ms_bool_features, self.ms_frequency_features,
            self.ms_count_features, self.chemical_features, self.annotations)
        features = list(itertools.chain(*map(make_features, feature_list)))
        #  features += [["Annotations", "Known"]]

        _names = ('campaign_id', 'protein_id', 'start', 'stop')
        _tuples = [(self.campaign_id, self.protein_id, *p.pos) for (p, c) in peptides]
        index = pd.MultiIndex.from_tuples(_tuples, names=_names)
        columns = pd.MultiIndex.from_tuples(features)
        types = {c: float for c in columns}
        # Annotations have different types
        #  types['Target', 'known'] = bool
        #  for annotation_name, annotation_dtype in self.annotation_dtypes.items():
        _iter = zip(self.annotations.features, self.annotations.dtypes)
        for annotation_name, annotation_dtype in _iter:
            types[self.annotations.type, annotation_name] = annotation_dtype
        for ms_bool_feature in self.ms_bool_features.features:
            types[self.ms_bool_features.type, ms_bool_feature] = bool

        df = pd.DataFrame(np.zeros((len(peptides), len(features))) * np.nan,
                          index=index, columns=columns).T
        lpvs = None
        if lpv_column:
            lpvs = self.lpv_creator.predict(2, True)
        for peptide, n_cluster in peptides:
            _index = (self.campaign_id, self.protein_id, peptide.start.pos, peptide.stop.pos)
            peptide_series = self._add_features_to_peptide_series(peptide, df.index, n_cluster,
                                                                  lpvs)
            df[_index] = peptide_series
        return df.T.astype(types)

    @classmethod
    def iterrows(cls, df):
        for id_tuple, data in df.iterrows():
            yield PeptideVariantId(*id_tuple[1:]), data

    # helper methods
    def _add_features_to_peptide_series(self, peptide, index, n_cluster=-1, lpvs=None):
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
        if lpvs is None:
            lpvs = set()
        i_start = peptide.start.index
        i_stop = peptide.stop.index

        # MS Delta
        series = pd.Series(np.zeros(len(index)) * np.nan, index=index)
        ms_int = self.ms_intensity_features.type
        series[ms_int, 'start'] = self.start_scores[i_start]
        series[ms_int, 'stop'] = self.stop_scores[i_stop]

        if 4 < len(peptide):
            penalty = SequenceRange(peptide.start + 1, peptide.stop - 1, validate=False)
            series[ms_int, 'penalty_start'] = self.start_scores[penalty.slice].sum()
            series[ms_int, 'penalty_stop'] = self.stop_scores[penalty.slice].sum()
        else:
            series[ms_int, 'penalty_start'] = series[ms_int, 'penalty_stop'] = 0

        # MS Bool
        b_obs, f_obs = self._calc_observed(peptide)
        series[self.ms_bool_features.type, "first"] = self.h_first[i_start]
        series[self.ms_bool_features.type, "last"] = self.h_last[i_stop]
        series[self.ms_bool_features.type, "observed"] = b_obs

        # MS Frequency
        # ptm weights
        # TODO: should it get extra penalties if there are PTM's between start and end?
        ms_freq = self.ms_frequency_features.type
        series[ms_freq, 'acetylation'] = self.ac_freq[i_start]
        series[ms_freq, 'amidation'] = self.am_freq[i_stop]

        series[ms_freq, 'start'] = self.h_start_freq[i_start]
        series[ms_freq, 'stop'] = self.h_stop_freq[i_stop]
        series[ms_freq, 'observed'] = f_obs
        series[ms_freq, 'sample'] = self.h_sample[peptide.slice].min()
        series[ms_freq, 'ladder'] = \
            self.h_ladder_start[i_start] * self.h_ladder_stop[i_stop]
        series[ms_freq, 'protein_coverage'] = self.protein_coverage
        series[ms_freq, 'cluster_coverage'] = self.cluster_coverage[n_cluster]

        # thise are good features, but there may be better ways to extract them
        series[ms_freq, 'bond'] = self.h_bond[self.get_bond_slice(peptide)].min()

        # MS Counts
        ms_count = self.ms_count_features.type
        series[ms_count, 'start'] = self.start_counts[peptide.start]
        series[ms_count, 'stop'] = self.stop_counts[peptide.stop]
        #  series[ms_count, 'ladder'] = \
        #      self.h_ladder_start[i_start] + self.h_ladder_stop[i_stop]

        ############################################################

        # Chemical
        sequence = self.protein_sequence[peptide.slice]
        peptide_features = GlobalDescriptor(sequence)

        is_amidated = series[ms_freq, 'amidation'] > 0.05
        peptide_features.calculate_all(amide=is_amidated)

        chem = self.chemical_features.type
        for i, name in enumerate(peptide_features.featurenames):
            if name in self.chemical_features.features:
                series[chem, name] = peptide_features.descriptor[0, i]

            eisenberg = PeptideDescriptor(sequence, 'eisenberg')
            eisenberg.calculate_moment()
            series[chem, 'eisenberg'] = eisenberg.descriptor.flatten()[0]

        # Annotations
        series[self.annotations.type, "Known"] = peptide in self.known_peptides
        #  series[self.annotations.type, "Type"] = peptide in self.known_peptides
        series[self.annotations.type, "Cluster"] = n_cluster
        series[self.annotations.type, "Sequence"] = peptide.seq
        series[self.annotations.type, "LPV"] = False  # TODO!

        series[self.annotations.type, "N Flanking"] = \
            self.get_nflanking_region(peptide.start, self.protein_sequence)
        series[self.annotations.type, "C Flanking"] = \
            self.get_cflanking_region(peptide.stop, self.protein_sequence)
        series[self.annotations.type, "LPV"] = peptide in lpvs
        if f_obs != 0:
            _pep_index = (slice(None), slice(None), peptide.start.pos, peptide.stop.pos)
            series[self.annotations.type, "Intensity"] = self.df.loc[_pep_index, :].sum().sum()
        return series

    def get_nflanking_region(self, start, protein_sequence, size=4):
        protein_padded = "_" * size + protein_sequence
        return SequenceRange(start, length=size, full_sequence=protein_padded).seq

    def get_cflanking_region(self, stop, protein_sequence, size=4):
        protein_padded = protein_sequence + "_" * size
        return SequenceRange(stop + 1, length=size, full_sequence=protein_padded).seq

    def _calc_observed(self, peptide) -> (bool, float):
        """
        returns:
            Is the peptide in the data (True/False)
            fraction of histogram explained peptide (float between 0 and 1)
        """
        if peptide not in self.upf_entries:
            return False, 0
        obs_area = self.upf_entries[peptide] * peptide.length / self.n_samples
        h_area = self.h[peptide.slice].sum()
        return True, obs_area / h_area

    def get_peptides_by_type(self, type_: str):
        peptide_types = {'valid': self.valid_peptides, 'fair': self.fair_peptides}
        if type_ not in peptide_types:
            raise ValueError(f"{type_} not in {peptide_types.keys()}")  # noqa
        return peptide_types[type_]

    @classmethod
    def calc_coverage(self, no_aa):
        return (no_aa.shape[0] - no_aa.sum()) / no_aa.shape[0]

    @classmethod
    def calcuate_cluster_coverage(cls, clusters, h_cluster, upf_entries):
        coverage = collections.defaultdict(int)
        for peptide in upf_entries.keys():
            n_cluster = h_cluster[peptide.start.index]
            coverage[n_cluster] += len(peptide) / len(clusters[n_cluster])
        return coverage

    ########################################
    # Histogram manipulation
    ########################################
    @classmethod
    def normalize_histogram(cls, histogram, bg_histogram):
        h = np.zeros(len(bg_histogram))
        not_zero = histogram != 0
        h[not_zero] = histogram[not_zero] / bg_histogram[not_zero]
        return h

    #  def fake_log10(self, data, imp_val=1):
    #      "normalizes data to imp_val, default is 1 as this will make all 0 stay 0"
    #      data = data.copy()
    #      # if a peptide > imputation score, it would get a 'negative start score!!!'
    #      data[data <= 10 ** imp_val] = 10 ** imp_val
    #      return np.log10(data)
    #
    #  def get_scores(self, histogram, imp_val):
    #      with np.errstate(divide='raise'):
    #          h10 = self.fake_log10(histogram, imp_val)
    #          #  self.h_overlap = np.zeros(self.length)
    #
    #          h10_padded = np.ones(self.length + 2) * imp_val
    #          h10_padded[1:-1] = h10
    #
    #          start_scores = h10 - h10_padded[:-2]
    #          start_scores[start_scores < 0] = 0
    #
    #          stop_scores = h10 - h10_padded[2:]
    #          stop_scores[stop_scores < 0] = 0
    #          return start_scores, stop_scores

    def _pad_array(self, array):
        padded = np.zeros(array.shape[0] + 2)
        padded[1:-1] = array
        return padded

    def get_terminal_histograms(self, histogram, histogram_start, histogram_stop):
        """
        intuition behind formular:
        a start/stop point is good if it is higher than the previous/next possition
        only count the "height" from peptides who are 'contenious', ie: subtract the height
        contribution from peptides who start/stop before the start and after the stop
        vissual aid:

        no overlap: peptide 2 start should not be penalized by peptide 1:
                               i
        peptide 1: ------------
        peptide 2:             --------------
        start    : -           -
        stop     :            -             -
        histogram: --------------------------
        start_{i}= h_{i} - h_{i-1} + h_{stop,i-1}
        start_{i}=     '-'     -    '-'  + '-'   = '-'  <-- good start place
                               i
        no overlap: peptide 2 start should be penalized by peptide 1:
        peptide 1: ----------------
        peptide 2:             --------------
        histogram: ------------====----------
        start    : -           -
        stop     :                -         -
        start_{i}= h_{i} - h_{i-1} + h_{stop,i-1}
        start =        '-'     -   '-'   +    ' ' = ' '  <-- bad start place

        note: we add a pseudo counnt of 1 to all thise, as this makes log10(data) zero or positive.
        returns 2 histograms: (log_{10}(start), log_{10}(stop))
        """

        histogram_padded = self._pad_array(histogram)
        histogram_start_padded = self._pad_array(histogram_start)
        histogram_stop_padded = self._pad_array(histogram_stop)

        #  #  start_{i} =   h_{start,i}   -        h_{i-1}        +        h_{stop,i-1}        + 1
        #  start_scores = histogram_start - histogram_padded[:-2] + histogram_stop_padded[:-2] + 1
        #  start_scores[start_scores < 1] = 1
        #  start_scores = np.log10(start_scores)
        #  #  stop_{i} =    h_{stop,i}  -        h_{i+1}       +       h_{start,i+1}        + 1
        #  stop_scores = histogram_stop - histogram_padded[2:] + histogram_start_padded[2:] + 1
        #  stop_scores[stop_scores < 1] = 1
        #  stop_scores = np.log10(stop_scores)

        #  start_{i} =   h_{start,i}   -        h_{i-1}        +        h_{stop,i-1}        + 1
        start_previous = np.log10(histogram_padded[:-2] - histogram_stop_padded[:-2] + 1)
        delta_start = np.log10(histogram + 1) - start_previous
        delta_start[delta_start < 0] = 0
        #  stop_{i} =    h_{stop,i}  -        h_{i+1}       +       h_{start,i+1}        + 1
        stop_subsequent = np.log10(histogram_padded[2:] - histogram_start_padded[2:] + 1)
        delta_stop = np.log10(histogram + 1) - stop_subsequent
        delta_stop[delta_stop < 0] = 0

        #  # debug!!!
        #  #  start_{i} =   h_{start,i}   -        h_{i-1}        +        h_{stop,i-1}        + 1
        #  start_previous2 = np.log10(histogram_padded[:-2] + 1)
        #  delta_start2 = np.log10(histogram + 1) - start_previous2
        #  delta_start2[delta_start2 < 0] = 0
        #  #  stop_{i} =    h_{stop,i}  -        h_{i+1}       +       h_{start,i+1}        + 1
        #  stop_subsequent2 = np.log10(histogram_padded[2:] + 1)
        #  delta_stop2 = np.log10(histogram + 1) - stop_subsequent2
        #  delta_stop2[delta_stop2 < 0] = 0
        #  if not all(delta_start == delta_start2):
        #      import colored_traceback.auto; import ipdb; ipdb.set_trace()  # noqa
        #  elif not all(delta_stop == delta_stop2):
        #      import colored_traceback.auto; import ipdb; ipdb.set_trace()  # noqa
        #  #  import colored_traceback.auto; import ipdb; ipdb.set_trace()  # noqa

        return delta_start, delta_stop

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
    def make_histograms(cls, df, length, n_samples, ladder_window=5):
        histogram = np.zeros(length)
        histogram_start = np.zeros(length)
        histogram_stop = np.zeros(length)
        histogram_ac = np.zeros(length)
        histogram_am = np.zeros(length)
        #  histogram_bonds = np.zeros(length - 1)

        for pep_var_id, peptide_series in cls.iterrows(df):
            p = SequenceRange(pep_var_id.start, pep_var_id.stop)
            intensity = peptide_series.sum() / n_samples
            if pep_var_id.mod_seq.startswith('_(ac)'):
                histogram_ac[p.start.index] += intensity
            if pep_var_id.mod_seq.endswith('_(am)'):
                histogram_am[p.stop.index] += intensity
            histogram_start[p.start.index] += intensity
            histogram_stop[p.stop.index] += intensity
            histogram[p.slice] += intensity

        bonds = np.stack((histogram[1:], histogram[:-1]))
        with np.errstate(invalid='ignore'):
            histogram_bonds = bonds.min(axis=0) / bonds.max(axis=0)

        first = (histogram == histogram_start) & (histogram != 0)
        last = (histogram == histogram_stop) & (histogram != 0)

        return (histogram, histogram_start, histogram_stop, histogram_ac, histogram_am,
                histogram_bonds, first, last)

    @classmethod
    def get_bond_slice(cls, peptide):
        """
        A peptide like this SequenceRange(10, 20), has a length of 11, but only 10 bonds, thus
        SequenceRange(10, 20).slice -> slice(9, 20)
        cls.get_bodn_slice(SeuqenceRange(10, 20) -> slice(9, 19)
        """
        return SequenceRange(peptide.start, peptide.stop - 1).slice

    #  @classmethod
    def make_sample_frequency_histogram(self, df):
        histogram_samples = pd.DataFrame(np.zeros((self.length, df.shape[1])),
                                         columns=df.columns)
        for pep_var_id, peptide_series in self.iterrows(df):
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
    def get_clusters(cls, h_cluster):
        clusters = {}
        for n_clust in range(1, h_cluster.max() + 1):
            cluster_indexes = np.where(h_cluster == n_clust)[0]
            clusters[n_clust] = SequenceRange.from_index(
                cluster_indexes[0], cluster_indexes[-1])
        return clusters

    @classmethod
    def count_ladders(cls, position_counts, h_cluster, clusters, ladder_window=10):
        """
        Returns the percentages of top +/- window_ladder around a possition_count

        thus if there are 5 peptides that stops at position 100
        and 10 peptides that stop within 10 of that position the that index of the returned array
        would be: 5 / (10 + 5) = 0.3333..
        thus close to 0 means loads of close starting positions, and 1 means only starting position
        """
        # TODO: ladders should take into account the number of start stops, IE
        # if 5 starts at the position and 10 peptides start 5 other places

        # 1 / (1 + 5) = 1/6  <--- how we do it in the code below
        #  counts = np.zeros(h_cluster.shape[0])
        #  for position in positions:
        #      counts[position.index] = 1
        #  h_ladder = np.zeros(h_cluster.shape[0])

        # 5 / (5 + 10) = 1/3 <--- ideal
        counts = np.zeros(h_cluster.shape[0])
        for position, count in position_counts.items():
            counts[position.index] = count
        h_ladder = np.zeros(h_cluster.shape[0])

        # ladders are pos +/- ladder_window, but has to stay within cluster boundaries
        #  for position in positions.items():
        for position, count in position_counts.items():
            n_cluster = h_cluster[position.index]
            ladder_start = max(clusters[n_cluster].start, position.pos - ladder_window)
            ladder_stop = min(clusters[n_cluster].stop, position.pos + ladder_window)
            ladder_range = SequenceRange(ladder_start, ladder_stop)
            #  h_ladder[position.index] = counts[ladder_range.slice].sum() - 1
            h_ladder[position.index] = count / counts[ladder_range.slice].sum()
        return h_ladder

    @classmethod
    def get_cluster_histogram(cls, no_aa):
        cluster = np.zeros(no_aa.shape, dtype=int)
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

    @classmethod
    def get_valid_positions(cls, peptides, h_cluster):
        valid_starts = {}
        valid_stops = {}
        #  for upf_entry in upf_entries:
        #      p = SequenceRange(upf_entry.start, upf_entry.end)
        for p in peptides:
            valid_starts[p.start] = h_cluster[p.start.index]
            valid_stops[p.stop] = h_cluster[p.stop.index]
        return valid_starts, valid_stops

    def get_possition_counts(cls, peptides):
        starts = collections.defaultdict(int)
        stops = collections.defaultdict(int)
        for p in peptides:
            starts[p.start] += 1
            stops[p.stop] += 1
        return dict(starts), dict(stops)

    @classmethod
    def get_valid_peptides(cls, valid_starts, valid_stops, protein_sequence):
        valid_peptides = {}
        for v_start, c_start in valid_starts.items():
            for v_stop, c_stop in valid_stops.items():
                if v_start < v_stop and c_start == c_stop:
                    p = SequenceRange(v_start, v_stop, full_sequence=protein_sequence)
                    valid_peptides[p] = c_start
        return valid_peptides


class LPVCreator:
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

    def __init__(self, peptides, protein_sequence, h_ac, h_am):
        self.peptides = peptides
        self.protein_sequence = protein_sequence
        self.h_ac = h_ac
        self.h_am = h_am

    ########################################
    # kelstrup algorithm
    ########################################
    def predict(self, min_overlap=2, ptm_flag=True):

        lpv_iter = self._iter_build_lpv(min_overlap)
        if not ptm_flag:
            return set(lpv_iter)
        else:
            return set(self._iter_split_ptm(lpv_iter))

    def _iter_build_lpv(self, min_overlap):
        # phase 1)
        # sorted reverse, because you can only pop from the end
        peptides = sorted(self.peptides, reverse=True)
        if len(peptides) == 0:
            return
        lpv = peptides.pop()
        while len(peptides) != 0:
            pep = peptides.pop()

            # Step 1) you are inside the peptide - ignore/delete -
            if pep in lpv:
                continue

            # Step 2) you are extending the peptide - extend -
            overlap = lpv.stop - pep.start + 1
            if overlap >= min_overlap:
                lpv = SequenceRange(lpv.start, pep.stop, full_sequence=self.protein_sequence)
                continue

            # no extension, no internal -> new lpv
            yield lpv
            lpv = pep
        yield lpv

    def _iter_split_ptm(self, lpv_iter):
        # warning this iter can return the same lpv twice (so convert to set!)
        # Phase 2)
        for lpv in lpv_iter:
            pos_array = np.arange(len(lpv)) + lpv.start.pos
            starts = set(pos_array[self.h_ac[lpv.slice] != 0]) | {lpv.start.pos}
            stops = set(pos_array[self.h_am[lpv.slice] != 0]) | {lpv.stop.pos}
            for start in starts:
                yield SequenceRange(start, lpv.stop, full_sequence=self.protein_sequence)
            for stop in stops:
                yield SequenceRange(lpv.start, stop, full_sequence=self.protein_sequence)
            yield lpv
