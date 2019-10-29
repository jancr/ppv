# core imports
import collections
from collections import abc
import copy
import sys
import typing

# 3rd party imports
import numpy as np
import tqdm
import pandas as pd

# local imports
from peputils import om, qc
from sequtils import SequenceRange
from peputils.proteome import fasta_to_protein_hash

from .split import Fold, XFold

Known = collections.namedtuple("Known", "protein_id", "start", "stop", "seq", "mod_seq", "type",
                               "full_name", "short_name")


################################################################################
# PeptideCollapser
################################################################################
class DatasetFeatureExtractor:
    def __init__(self,
                 df: pd.DataFrame,
                 protein_sequences: typing.Union[[str, typing.Dict[str, str]]],
                 known_peptides: typing.Union[str, None, typing.Dict[str, set]] = None):

        self.df = pd.DataFrame()
        self.know = self._resolve_known_peptides(known_peptides)
        self.protein_sequences = self._resolve_protein_sequences(protein_sequences)

    # __init__ helpers
    def _resolve_known_peptides(self, known_peptides):
        if known_peptides is None:
            return {}
        elif isinstance(known_peptides, abc.Mapping):
            return known_peptides
        elif isinstance(known_peptides, str):
            self.known = self.get_known_peptides(known_peptides)
        error = "known_peptides must be of type None, str or Mapping, not {}"
        raise ValueError(error.format(type(known_peptides)))

    @staticmethod
    def get_known_peptides(known_file):
        #  known_peptides = collections.defaultdict(dict)
        known_peptides = collections.defaultdict(set)
        with open(known_file) as known_file:
            known_file.readline()  # skip header
            for line in known_file:
                known = Known(line.rstrip('\r\n').split('\t'))
                if known.type in ('peptide', 'propeptide'):
                    peptide = SequenceRange(int(known.start), int(known.stop), seq=known.seq)
                    known_peptides[known.protein_id].add(peptide)
                    #  known_peptides[known.protein_id][p] = known.short_name.split(',')[0]
        return dict(known_peptides)


# INSPIRATION:
# INSPIRATION:
# INSPIRATION:
# INSPIRATION:
class PeptideCollapser:
    #  def __init__(self, upf_files, protein_seqs, organism=None,
    #               ortholog_organism=None, known=None, meta_data=(), normalize=True):
    def __init__(self, protein_seqs, upf_files=(), meta_data=(), organism=None,
                 ortholog_organism=None, known=None, normalize=True):
        # init instance variables
        self.organism = organism
        self.ortholog_organism = ortholog_organism

        if isinstance(protein_seqs, str):
            self.protein_seqs = fasta_to_protein_hash(protein_seqs)
        else:
            self.protein_seqs = protein_seqs

            #  self.priors, self.average_of_priors, self.average_prior = self.get_priors().

        if known is not None:
            self.known_peptides = self.get_known_peptides(known)

        # set upf instance variables
        if isinstance(upf_files, (str, bytes)):
            upf_files = (upf_files,)
            if isinstance(meta_data, (str, bytes, om.MetaData)):
                meta_data = (meta_data,)
            elif isinstance(meta_data, abc.Iterabe) and len(meta_data) == 0:
                meta_data = tuple([None] * len(upf_files))
        self.n_upf_files = len(upf_files)

        self.upf_dfs = tuple(self.parse_upf_dataset(upf_file, meta, normalize)
                             for upf_file, meta in zip(upf_files, meta_data))
        #  if meta_data is None:
        #      meta_data = [None] * len(upf_files)
        #  self.upf_peptides = [self.get_upf_peptides(upf_file, meta_file)
        #                      for (upf_file, meta_file) in zip(upf_files, meta_data)]

        self.peptide_scorers = self.get_peptide_scorers()

    def parse_upf_dataset(self, upf_file, meta_data, normalize=True, zero_missing=True):
        upf_df = om.load_upf_as_df(upf_file, meta_data, pep_id=False, zero_missing=zero_missing)
        if normalize:
            upf_df = 10 ** qc.df_normalize(np.log10(upf_df))

        df = {}
        for protein_id in upf_df.index.levels[0]:
            df[protein_id] = upf_df.loc[protein_id]
        return df

    def add_upf_dataset(self, upf_file, meta_data, normalize=True, zero_missing=True):
        self.upf_dfs.append(self.parse_upf_dataset(upf_file, meta_data, normalize, zero_missing))

    ########################################
    # PICKLE api:
    ########################################
    def __getstate__(self):
        state = copy.copy(self.__dict__)
        del state['peptide_scorers']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.peptide_scorers = self.get_peptide_scorers()

    #  def get_peptide_scorer_class(self):
    #      if self.organism is not None and self.ortholog_organism is not None:
    #          if (self.organism, self.ortholog_organism) == (10090, 9606):
    #              return PeptideScorerMouseToHuman
    #          elif (self.organism, self.ortholog_organism) == (9606, 10090):
    #              return PeptideScorerHumanToMouse
    #          elif (self.organism, self.ortholog_organism) == (9544, 9606):
    #              return PeptideScorerMacaqueToHuman
    #          raise ValueError("No class exists for that species combination"
    #                           "and I am to stupid to make it work generically")
    #          #  return PeptideScorerCons
    #          #  class PeptideScorerCons(PeptideScorer):
    #          #      pass
    #          #  PeptideScorerCons._ss = SequenceScorer(4)
    #          #  PeptideScorerCons.organism = self.organism
    #          #  PeptideScorerCons.ortholog = self.ortholog_organism
    #          #  PeptideScorerCons.nog_level = 'maNOG'
    #          #
    #          #  return PeptideScorerCons
    #      return PeptideScorer

    def get_peptide_scorers(self):
        #  PS = self.get_peptide_scorer_class()
        peptide_scorers = [{} for i in range(self.n_upf_files)]
        for upf_index, _peptide_scorers in enumerate(peptide_scorers):
            #  for protein_id, upf_entries in self.upf_peptides[upf_index].items():
            for protein_id, upf_df in self.upf_dfs[upf_index].items():
                known = set()
                if hasattr(self, 'known_peptides') and protein_id in self.known_peptides:
                    known = set(self.known_peptides[protein_id])
                if protein_id in self.protein_seqs:  # NHP hack!!
                    ps = PeptideScorer(protein_id, self.protein_seqs[protein_id], upf_df,
                                       known_peptides=known)
                else:
                    print('NHP lost:', protein_id)
                _peptide_scorers[protein_id] = ps
        return peptide_scorers

    def get_upf_peptides(self, upf_file, meta_file=None):
        peptides = {}
        for upf_entry in om.UPFEntry.iter_file(upf_file, meta_file):
            if upf_entry.valid_intensity:
                peptides.setdefault(upf_entry.protein_id, []).append(upf_entry)
        return peptides

    @staticmethod
    def get_known_peptides(known_file):
        known_peptides = collections.defaultdict(dict)
        with open(known_file) as known_file:
            known_file.readline()  # skip header
            for line in known_file:
                (protein_id, begin, end, seq, mod_seq, type_, full_name, short_name) = \
                    line.rstrip('\r\n').split('\t')
                if type_ == 'peptide' or type_ == 'propeptide':
                    p = SequenceRange(int(begin), int(end))
                    known_peptides[protein_id][p] = short_name.split(',')[0]
        return dict(known_peptides)

    def write_ppvs(self, w_start, w_end, w_term, w_ac=0, w_am=0, w_length=0, imp_val=0,
                   w_obs=0, max_overlap=None, file=sys.stdout, mod_threshold=0.01, n_cpus=1,
                   predictions_per_amino_acid=0.1, cap_predictions_to_observations=True):
        p = Parameters(w_start, w_end, w_term, w_ac, w_am, w_length, imp_val, w_obs)
        ppvs = self.predict_peptides(p, max_overlap=max_overlap, n_cpus=n_cpus)
        if isinstance(file, str):
            file = open(file, 'w')

        parameters = "w_start={},w_end={},w_term={},w_ac={},w_am={},w_length={},imp_val={}"
        parameters = parameters.format(w_start, w_end, w_term, w_ac, w_am, w_length, imp_val)
        print("protein_id", "begin", "end", "seq", "mod_seq", "score",
              "rank", "parameters", "version", file=file, sep='\t', end='\n')
        for upf_index, upf_ppvs in enumerate(ppvs):
            for protein_id, ppv_peptides in upf_ppvs.items():
                ppvs_sorted = sorted(ppv_peptides.items(), key=lambda x: -x[-1])
                ps = self.peptide_scorers[upf_index][protein_id]
                n_predictions = round(len(ps.protein_seq) * predictions_per_amino_acid)
                if cap_predictions_to_observations:
                    n_predictions = min(n_predictions, len(ps.upf_entries))
                _iter = enumerate(ppvs_sorted[:n_predictions], 1)
                for rank, (p, score) in _iter:
                    seq = ps.protein_seq[p.get_slice()]
                    if ps.ac_freq[p.start_index] >= mod_threshold:
                        mod_seq = '_(ac){}_'.format(seq)
                    else:
                        mod_seq = '_{}_'.format(seq)

                    if ps.am_freq[p.end_index] >= mod_threshold:
                        mod_seq += '(am)'
                    print(protein_id, p.start_pos, p.end_pos, seq, mod_seq,
                          '{:.3f}'.format(score), rank, parameters, __version__,
                          file=file, sep='\t', end='\n')

    def predict_peptides(self, parameters, max_overlap=None, train=False, folds=None, n_cpus=1,
                         mode='ppv'):
        all_collapsed_peptides = []
        for upf_index in range(self.n_upf_files):
            ########################################
            # subset data if we are training
            ########################################
            upf_peptides = self.upf_dfs[upf_index]
            if train:
                # only predict on non-holdout
                if folds is not None:
                    protein_ids = set()
                    if isinstance(folds, Fold):
                        folds = (folds,)
                    for fold in folds:
                        protein_ids |= fold.proteins.keys()
                    protein_ids &= upf_peptides.keys()
                else:  # all knowns in data
                    protein_ids = set(self.known_peptides.keys()) & set(upf_peptides.keys())
            else:
                protein_ids = set(upf_peptides.keys())

            ########################################
            # run predictions
            ########################################
            def get_collapsed(peptide_scorer):
                if mode == 'ppv':
                    return peptide_scorer.get_obsered_predictions, ()
                elif mode == 'observed':
                    return peptide_scorer.get_all_predictions, (max_overlap,)

            collapsed_peptides = {}
            total = len(protein_ids)
            if n_cpus == 1:
                for i, protein_id in tqdm.tqdm(enumerate(protein_ids), "ppv predict", total):
                    peptide_scorer = self.peptide_scorers[upf_index][protein_id]
                    peptide_scorer.change_params(parameters)
                    prediction_method, args = get_collapsed(peptide_scorer)
                    collapsed_peptides[protein_id] = prediction_method(*args)
            else:
                def _future_done(future):
                    future.pbar.update(len(future.peptide_scorer.valid_peptides))
                    collapsed_peptides[future.peptide_scorer.protein_id] = future.result()

                with concurrent.futures.ProcessPoolExecutor(n_cpus) as exe:
                    peptide_scorers = sorted(self.peptide_scorers[upf_index].values(),
                                             key=lambda p: -len(p.valid_peptides))
                    pbar = tqdm.tqdm(desc="ppv predict (largest chunks first)",
                                     total=sum(len(p.valid_peptides) for p in peptide_scorers))
                    for peptide_scorer in peptide_scorers:
                        peptide_scorer.change_params(parameters)
                        prediction_method, args = get_collapsed(peptide_scorer)
                        future = exe.submit(prediction_method, *args)
                        future.peptide_scorer = peptide_scorer
                        future.pbar = pbar
                        future.add_done_callback(_future_done)
            all_collapsed_peptides.append(collapsed_peptides)
        return all_collapsed_peptides

    def get_priors(self):
        # TODO: code has not been updated to handle multiple upf datasets
        # TODO: code has not been updated to handle multiple upf datasets
        # TODO: this code uses the old upf_peptides instead of upf_dfs
        priors = {}
        total_all = hit_all = 0
        # for protein_id, peptides in self.upf_peptides.items():
        for protein_id, upf_entries in self.upf_peptides.items():
            if protein_id not in self.known_peptides:
                continue
            unique_peptides = set()
            # for (start_index, end_index, pep_mod_seq, intensity) in peptides:
            for upf_entry in upf_entries:
                unique_peptides.add((upf_entry.start, upf_entry.end))
            hit = total = 0
            for peptide_loc in unique_peptides:
                if peptide_loc in self.known_peptides[protein_id]:
                    hit += 1
                total += 1
            hit_all += hit
            total_all += total
            priors[protein_id] = hit / total

        average_of_priors = sum(priors.values()) / len(priors)
        average_prior = hit_all / total_all
        return priors, average_of_priors, average_prior

    def _get_folds_from_state(self, xfolds, state):
        if state is None:
            return None
        if state == 'train':
            return xfolds.get_training_folds()
        elif state == 'test':
            return xfolds.get_holdout_fold()
        elif state == 'validate':
            return xfolds.get_validation_fold()
        #  state_lookup = {
            #  'train': xfolds.get_training_folds,
            #  'test': xfolds.get_holdout_fold,
            #  'validate': xfolds.get_validation_fold
        #  }
        #  if state in state_lookup:
            #  return state_lookup[state]()
        raise ValueError("state has to be 'train', 'test', 'validate' or None")

    def _eval(self, x, xfolds=None, state=None,
              eval_fun=Eval.squared_pos_weighted, verbose=False):
        folds = self._get_folds_from_state(xfolds, state)
        print(x)
        parameters = Parameters(*x)
        all_predictions = self.predict_peptides(parameters, train=True, folds=folds)

        weight_sum = 0
        eval_sum = 0
        dtype = [('score', float), ('pos', bool)]
        for upf_index, upf_prediction in enumerate(all_predictions):
            for (protein, peptides) in upf_prediction.items():
                n_positive = len(self.peptide_scorers[upf_index][protein].findable)
                if n_positive == 0 or n_positive == len(peptides):
                    continue

                scores = np.empty(len(peptides), dtype=dtype)
                for j, (pos, score) in enumerate(peptides.items()):
                    is_known = pos in self.known_peptides[protein]
                    scores[j] = (score, is_known)
                eval_score, w = eval_fun(scores)
                assert n_positive == w, "n_positive={}, w={}, i={}".format(n_positive, w,
                                                                           upf_index)
                weight_sum += w
                eval_sum += eval_score

        if verbose:
            print(parameters, (eval_sum / weight_sum))
        return (eval_sum / weight_sum)  # we need to find local minima

    def generate_xfolds(self, n_folds: int, validation=None):
        # only count known peptides who are findable in the dataset...
        # thus a known peptide who does not share a start and end with a upf_peptid does not count
        # just like a "negative" that does not have a upf_start and end are not considered either
        n_known_peptides = collections.defaultdict(int)

        # self.findable = [{} for _ in range(self.n_upf_files)]
        for protein_id, known_peptides in self.known_peptides.items():
            for i, peptide_scorers in enumerate(self.peptide_scorers):
                if protein_id in peptide_scorers:
                    ps = peptide_scorers[protein_id]
                    if len(ps.findable) != 0 and len(ps.findable) != len(ps.valid_peptides):
                        n_known_peptides[protein_id] += len(ps.findable)

        xfold = XFold(n_folds, dict(n_known_peptides), validation)
        return xfold

    def train(self, w_start, w_end, w_term, w_ac, w_am, w_length, imp_val, w_obs,
              folds=1, validation=False, finish=True, verbose=False,
              score_fun=Eval.squared_pos_weighted, vs_kelstrup=True):
        """
        ## Peptide Prediction ##
        1. Split in [folds] folds, 5 is very standard
        - fold 1-4, for xfold validation with rotation
        - fold 5 to evaluate vs Kelstrup
        2. exclude known peptides (from 1-4) who are "unfindable"
        - "unfindable" peptides have "no valid_start" or "no valid_end"
        - Now your training space has been reduced to the subset that can be found by
        the algorithm... This score will naturally be to high compared to the more
        honest approach... But it will be consistent between folds, where the number
        of unfindable were not guaranteed to be uniform, thus the "hold out" set
        should now reflect the extent of over training rather than how lucky the
        split was.
        3. Make all prediction, all starts to all ends
        - AUC or MCC are both good and valid functions to evaluate performance,
          the problem is that they are invariant to possition, considering a
          dataset with 4 known peptides and 996 combinations of "false peptides":
            - for MCC, if you are considering the 10 highest scoring as true and all else as false
              then
                1. you will get a terrible score because at best you will get 40% TP
                2. the list (1,2,11,12) is equivalent to the list (9,10, 999, 1000)
            - The AUC of the following are equivalen:
                - (1,2,999,1000) and (499,500,501,502), but any experimentalist
                  will perfere the former because they will only consider top 20 ish
            - one solution is to use WROC (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4129959/)
                - but "Condition (c) requires that one class is not dominating in the whole sample"
                  :(
            - Thus we have opted to use the square of the reverse index...
                - thus nr 1 will give 1000^2, nr 2 999^2 etc, then we normalize with the max
                  possible score
                Thus the AUC example will now give:
                    formular: score = (obs_score - min_score) / (max_score - min_score)
                    max_score = 1000^2 + 999^2 + 998^2 + 997^2
                    min_score = 1^2 + 2^2 + 3^2 + 4^2
                    (1000^2+999^2+2^2+1^2 - min_score) / (max_score - min_score) = 0.5 (ish)
                    (499^2+500^2+501^2+502 - min_score) / (max_score - min_score) = 0.25 (ish)
                thus giving the one where 2 are in top 20 twice the score of the one with 0 in top
                20!
        4. Use fold 5 to compare to Kelstrup, we have to use MCC here as Kelstrups
        algorithm is binary. We will handicap ourselves by predicting the same number
        of peptides as him, to be 100% comparable.
        """

        param_grid = ParameterGrid(w_start, w_end, w_term, w_ac, w_am, w_length, imp_val, w_obs)
        if folds == 1:
            def _eval(x):
                return -self._eval(x, verbose=verbose, eval_fun=score_fun)
            model_param = sp.optimize.brute(_eval, param_grid, finish=finish)
            return model_param

        xfolds = self.generate_xfolds(folds)

        model_params = []
        model_scores = []
        mccs = []
        if validation:
            xfolds.set_validation(folds - 1)

        for fold in xfolds:
            if fold.is_validation_fold():
                continue
            xfolds.set_holdout(fold.index)

            def brute_eval(x):
                return -self._eval(x, xfolds=xfolds, state='train', verbose=verbose,
                                   eval_fun=score_fun)
            if verbose:
                print(" -- hold out:", fold.index)

            model_param = sp.optimize.brute(brute_eval, param_grid, finish=finish)
            model_param = Parameters(*model_param)
            #    full_output=True) #  finish=optimize.fmin)

            train_score = self._eval(model_param, xfolds, 'train', score_fun, False)
            test_score = self._eval(model_param, xfolds, 'test', score_fun, False)
            model_params.append(model_param)
            model_scores.append((test_score, train_score))

            # train kelstrup before we mutate xfolds!
            if vs_kelstrup:
                mccs.append(self.vs_kelstrup(model_param, xfolds, 'test'))
        return (model_params, model_scores, xfolds, mccs)

    def print_eval_table(self, xfolds, model_params, model_scores, mccs=(), file=sys.stdout):
        if isinstance(file, str):
            file = open(file, 'w')
        for i in range(len(xfolds)):
            print('#fold {}'.format(i), xfolds.folds[i], file=file, sep='\t', end='\n')
        print(file=file, end='\n')
        print('holdout', 'train', 'test', 'test (mcc)', 'test kelstrup (mcc)',
              sep='\t', end='\t', file=file)
        print(*_parameter_fields, sep='\t', end='\n', file=file)

        _iter = enumerate(zip(model_scores, model_params))
        for holdout_index, ((test_score, train_score), parameters) in _iter:
            pred_mcc, kelstrup_mcc = mccs[holdout_index]
            parameter_dict = parameters._asdict()
            params = (parameter_dict[name] for name in _parameter_fields)
            print(holdout_index, train_score, test_score, pred_mcc, kelstrup_mcc,
                  *params, file=file, sep='\t', end='\n')

        n_folds = sum(1 for fold in xfolds.folds if not fold.is_validation_fold())
        avg_pred_mcc = sum(ppv_mcc for ppv_mcc, kelstrup_mcc in mccs) / n_folds
        avg_kelstrup_mcc = sum(kelstrup_mcc for ppv_mcc, kelstrup_mcc in mccs) / n_folds
        avg_train_score = sum(train for test, train in model_scores) / n_folds
        avg_test_score = sum(test for test, train in model_scores) / n_folds

        avg_params = Parameters(*(sum(p) / n_folds for p in zip(*model_params)))
        avg_param_dict = avg_params._asdict()
        avg_params = (avg_param_dict[name] for name in _parameter_fields)
        print('average', avg_train_score, avg_test_score, avg_pred_mcc, avg_kelstrup_mcc,
              *avg_params, file=file, sep='\t', end='\n')

    def print_eval_summary(self, xfolds, model_params, model_scores, mccs=()):
        # summary statistics
        print(model_scores, model_params, mccs)
        _iter = enumerate(zip(model_scores, model_params))
        for holdout_index, ((test_score, train_score), p) in _iter:
            print(" - evaluate fold {}".format(holdout_index))
            print("   = proteins:", xfolds.folds[holdout_index])
            print("   = params:", p)
            print("   = train: {}".format(train_score))
            print("   = test: {}".format(test_score))
            if len(mccs) != 0:
                pred_mcc, kelstrup_mcc = mccs[holdout_index]
                print("   = test mcc (our ppvs): {}".format(pred_mcc))
                print("   = test mcc (kelstrup): {}".format(kelstrup_mcc))

        # should probbably weight params by the score so the parameters that are best contribute
        # more :D
        avg_params = [sum(p) / (len(xfolds) - 1) for p in zip(*model_params)]
        if xfolds.validation_fold:
            validation_score = self._eval(avg_params, xfolds, 'validate', verbose=False)
            print(" - Validation")
            print("   = proteins:", xfolds.get_validation_fold())
            print("   = params:", avg_params)
            print("   = score: {}".format(validation_score))
            if len(mccs) != 0:
                pred_mcc, kelstrup_mcc = self.vs_kelstrup(avg_params, xfolds, 'validate')
                print("   = test mcc (our ppvs): {}".format(pred_mcc))
                print("   = test mcc (kelstrup): {}".format(kelstrup_mcc))

    def change_params(self, parameters):
        parameters = Parameters.to_param(parameters)
        for peptide_scorers in self.peptide_scorers:
            for peptide_scorer in peptide_scorers.values():
                peptide_scorer.change_params(parameters)

    def vs_kelstrup(self, params, xfolds, state):
        fold = self._get_folds_from_state(xfolds, state)
        ppv_prediction = self.predict_peptides(params, train=True, folds=fold)

        dtype = [('score', float), ('pos', bool)]
        k_mcc_sum = p_mcc_sum = 0
        k_mcc_list, p_mcc_list = [], []
        weights = 0
        for protein_id in fold.proteins:
            for upf_index in range(self.n_upf_files):
                if protein_id not in self.peptide_scorers[upf_index]:
                    continue
                k_pred = self.peptide_scorers[upf_index][protein_id].predict_kelstrup(
                    min_overlap=2, ptm_flag=True)
                p_pred = ppv_prediction[upf_index][protein_id]
                p_scores = np.empty(len(p_pred), dtype=dtype)

                n_known = 0
                n_total = len(p_pred)
                known_peptides = self.known_peptides[protein_id]
                for j, (pos, score) in enumerate(p_pred.items()):
                    is_known = pos in known_peptides
                    n_known += is_known
                    p_scores[j] = (score, is_known)
                if n_known == 0 or n_known == n_total:
                    continue
                weights += n_known

                k_tp = k_fp = 0
                for pos in k_pred:
                    if pos in known_peptides:
                        k_tp += 1
                    else:
                        k_fp += 1

                k_fn = n_known - k_tp
                k_tn = n_total - k_tp - k_fp - k_fn
                k_mcc = Eval.mcc(k_tp, k_tn, k_fp, k_fn) * n_known
                # print(' -- ', protein_id)
                # print('k mcc', k_mcc, n_known)
                # print("kelstrup param tp={}, fp={}, tn={}, fn={}".format(k_tp, k_fp, k_tn, k_fn))

                p_mcc, _ = Eval.mcc_weighted(p_scores, k_tp + k_fp)
                # print('v mcc', p_mcc, _)
                # print(v_scores[:10])
                p_mcc_sum += p_mcc
                k_mcc_sum += k_mcc
                # print()

                k_mcc_list.append((k_mcc, n_known))
                p_mcc_list.append((p_mcc, _))

        kelstrup_mcc = k_mcc_sum / weights
        prediction_mcc = p_mcc_sum / weights
        # print('weights', weights)
        # print("kelstrup MCC:   ", kelstrup_mcc)
        # print(" - ", k_mcc_list)
        # print("validation MCC: ", prediction_mcc)
        # print(" - ", p_mcc_list)
        return prediction_mcc, kelstrup_mcc

    def evaluate_prediction(self, pred_lpv, *args, **kwargs):
        score_bins = collections.defaultdict(list)
        for protein, lpvs in pred_lpv.items():
            for location, score in lpvs.items():
                is_known = location in self.known_peptides[protein]
                score_bins[int(score)].append(is_known)
        for score, hits in sorted(score_bins.items(), reverse=True):
            print("score: {} -> {}".format(score, sum(hits) / len(hits)))
