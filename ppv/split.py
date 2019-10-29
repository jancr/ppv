
# core imports
import math

__slots__ = ('Fold', 'XFold')


################################################################################
# XFold and Fold class
################################################################################
class Fold():
    """Class that represend each of the folds when doing X-fold evaluation"""
    def __init__(self, peptide_goal, protein_goal, fold_index=0,
                 holdout_fold=False, validation_fold=False):
        self.peptide_count = 0
        self.protein_count = 0
        self.proteins = {}
        self.index = fold_index

        self.peptide_goal = peptide_goal
        self.protein_goal = protein_goal
        self.protein_goal_high = math.ceil(protein_goal)
        self.protein_goal_low = math.floor(protein_goal)

        self.validation_fold = validation_fold
        self.holdout_fold = holdout_fold

    def perfect_protein(self):
        return abs(self.protein_count - self.protein_goal) < 1

    def perfect_peptide(self):
        return abs(self.peptide_count - self.peptide_goal) < 1

    def is_perfect(self):
        return self.perfect_peptide() and self.perfect_protein()

    def add(self, protein_id, peptide_count):
        if protein_id in self.proteins:
            raise KeyError("{} is already in this fold!".format(protein_id))
        self.proteins[protein_id] = peptide_count
        self.peptide_count += peptide_count
        self.protein_count += 1

    def remove(self, protein_id):
        peptide_count = self.proteins[protein_id]
        self.peptide_count -= peptide_count
        self.protein_count -= 1
        del self.proteins[protein_id]
        return peptide_count

    def is_holdout_fold(self):
        return self.holdout_fold

    def is_validation_fold(self):
        return self.validation_fold

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        _self = (self.peptide_count, self.protein_count, self.proteins)
        _other = (other.peptide_count, other.protein_count, other.proteins)
        return _self == _other

    def __str__(self):
        return ' '.join(map(str, (self.protein_count, self.peptide_count, self.proteins)))

    def __repr__(self):
        return self.__str__()


class XFold():
    """XFold class that perform balanced split by putting all peptides from
    one protein into the same bin.
    """
    def __init__(self, n_folds: int, positives, holdout_fold=None, validation_fold=None, run=True):
        self.n_folds = n_folds
        self.positives = positives
        self.validation_fold = validation_fold
        self.holdout_fold = holdout_fold

        self.positives = dict(positives)
        if len(positives) == 0:
            raise ValueError("The dataset has no known peptides :(")

        peptide_goal = sum(positives.values()) / n_folds
        protein_goal = len(positives) / n_folds

        self.folds = []
        for i in range(n_folds):
            fold = Fold(peptide_goal, protein_goal, i,
                        holdout_fold=i == holdout_fold,
                        validation_fold=i == validation_fold)
            self.folds.append(fold)

        if run:
            self.distribute(positives)
            self.trade()

    def __len__(self):
        return self.n_folds

    def __iter__(self):
        for fold in self.folds:
            yield fold

    def is_perfect(self):
        return all((f.is_perfect() for f in self.folds))

    @classmethod
    def move_closest(cls, fold_high, fold_low):
        if fold_high.perfect_protein() and fold_low.perfect_protein():
            cls._swap_two(fold_high, fold_low)
            # swap two proteins
        else:
            cls._move_one(fold_high, fold_low)
            # move 1 protein from high to low

    @classmethod
    def _swap_two(cls, fold_high, fold_low):
        #  this code is trivial to vectorize...
        #  but it is aready super confusing and not the rate limiting step!
        skip = set()
        smallest_diff = 999999
        for (high_id, high_count) in fold_high.proteins.items():
            if high_count in skip:
                continue
            skip.add(high_count)
            target_count = fold_low.peptide_count - fold_high.peptide_goal + high_count
            low_id, low_count = cls._find_one(fold_low, target_count)
            new_diff = abs(target_count - low_count)
            if new_diff < smallest_diff:
                smallest_diff = new_diff
                best_low_id = low_id
                best_high_id = high_id

        high_count = fold_high.remove(best_high_id)
        fold_low.add(best_high_id, high_count)

        low_count = fold_low.remove(best_low_id)
        fold_high.add(best_low_id, low_count)

    def _find_one(fold, target):
        count_iter = iter(fold.proteins.items())
        best_id, best_count = next(count_iter)
        for protein_id, peptide_count in count_iter:
            if abs(peptide_count - target) < abs(best_count - target):
                best_count = peptide_count
                best_id = protein_id
        return best_id, best_count

    @classmethod
    def _move_one(cls, fold_high, fold_low):
        target_count = fold_high.peptide_goal - fold_low.peptide_count
        best_id, best_count = cls._find_one(fold_high, target_count)
        fold_low.add(best_id, best_count)
        fold_high.remove(best_id)

    @staticmethod
    def _distribute_key(fold):
        """the mosth "worthy" fold is the one with fewest peptides and most
        proteins - index is part of the key to make the bins more "stable"
        between runs"""
        # index added to min key to make the folds always yield the same folds
        return fold.peptide_count, -fold.protein_count, fold.index

    def distribute(self, positives):
        """iteratively give the protein with the most peptides to the fold
        with the fewest peptides"""
        _sorted_data = sorted(positives.items(), key=lambda x: -x[1])
        for protein_id, n_peptides in _sorted_data:
            fold = min(self.folds, key=self._distribute_key)
            fold.add(protein_id, n_peptides)

    @staticmethod
    def _trade_key(fold):
        return fold.protein_count, fold.peptide_count

    def trade(self, max_iterations=100):
        """iteratively give a protein from the fold with most proteins,
        to the fold with the fewest"""
        for i in range(max_iterations):
            if self.is_perfect():
                break
            high_fold = max(self.folds, key=self._trade_key)
            low_fold = min(self.folds, key=self._trade_key)
            self.move_closest(high_fold, low_fold)
        else:  # no break
            raise RuntimeError("The fold-split did not converge")

    def set_holdout(self, holdout_index):
        """used to rotate fold under x-fold validation"""

        for fold in self.folds:
            fold.holdout_fold = fold.index == holdout_index
        self.holdout_fold = holdout_index

    def set_validation(self, validation_index):
        """this is used when developing, to ensure I do not touch
        the test set untill I am done developing"""

        for fold in self.folds:
            fold.validation_fold = fold.index == validation_index
        self.validation_fold = validation_index

    def iter_training_folds(self):
        for fold in self.folds:
            if not fold.is_validation_fold() and not fold.is_holdout_fold():
                yield fold

    def get_training_folds(self):
        return list(self.iter_training_folds())

    def get_validation_fold(self):
        if self.validation_fold is None:
            raise ValueError("No Validation Fold")
        return self.folds[self.validation_fold]

    def get_holdout_fold(self):
        return self.folds[self.holdout_fold]
