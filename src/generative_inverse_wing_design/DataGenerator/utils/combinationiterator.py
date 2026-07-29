"""Contains CombinationIterator class to generate permutations with repetitions."""

from itertools import product
from typing import Optional

import aerosandbox.numpy as np


class CombinationIterator:
    """Iterator class that generates all combinations with repetitions
      for given number of samples (number of combined classes) and
      the number of classes (distinct values).
    Args:
        n_samples: Number of samples.
        n_classes: Number of distinct classes (values) for each parameter.
        n_combinations: If specified, limits the number of combinations generated.
        tip_family: If specified, the last sample will always be this value.
        done_set: A set of already generated combinations to skip (skipped are not included in n_comb).
    Returns:
        An iterator that yields np.ndarray of shape (n_samples, ) for each combination.
        Note it is recomended to use n_combinations as a multiple of n_classes ** n_samples
        to have all combinations equally represented.
    """

    def __init__(self, n_samples: int,
                 n_classes: int,
                 n_combinations: Optional[int] = None,
                 tip_family: Optional[int] = None,
                 done_set: Optional[set] = None):

        if tip_family is not None:
            assert isinstance(tip_family, int) and 0 <= tip_family < n_classes, \
                "tip_family must be an integer between 0 and n_classes - 1."

            n_samples -= 1  # Reduce samples by one as tip is fixed

        if n_combinations is not None:
            assert isinstance(n_combinations, int) and n_combinations > 0,\
                "n_combinations must be a positive integer."
        else:
            n_combinations = n_classes ** n_samples  # All combinations

        if done_set is not None:
            self.done_combinations: set = done_set
            n_combinations = n_combinations - len(done_set)
        else:
            self.done_combinations: set = set()

        self._n_samples: int = n_samples
        self._n_classes: int = n_classes
        self._n_combinations: int = n_combinations
        self._tip_family: Optional[int] = tip_family
        self._count: int = 0
        # Calculate number of cycles needed to reach n_combinations rounded up
        self.n_cycles: float = (n_combinations/(n_classes ** n_samples))
        self._combinations = product(range(n_classes), repeat=n_samples)
        # product is a generator and returns the next combination on each iteration
        self._current_iter = iter(self._combinations)
        if not self.n_cycles.is_integer():
            print(
                f"Warning: n_combinations ({n_combinations}) is not a multiple of "
                f"n_classes ** n_samples ({n_classes ** n_samples}). "
                f"This may lead to uneven representation of combinations."
            )

    def __iter__(self):
        return self

    def __next__(self):
        while self._count <= self._n_combinations:
            try:
                result = np.array(next(self._current_iter), dtype=int)
            except StopIteration:
                # Restart the iterator if needed (for more than one cycle)
                self._combinations = product(range(self._n_classes), repeat=self._n_samples)
                self._current_iter = iter(self._combinations)
                result = np.array(next(self._current_iter), dtype=int)

            if self._tip_family is not None:
                result = np.append(result, self._tip_family)

            if tuple(result.tolist()) in self.done_combinations:
                continue

            self._count += 1
            return result

        raise StopIteration

    def __len__(self):
        # Total number of combinations times the number of cycles
        return self._n_combinations

def main():
    n_samples = 5
    n_classes = 14
    comb_gen = CombinationIterator(n_samples, n_classes, tip_family=2)
    print(f"Generating combinations: {comb_gen._n_combinations} == {n_classes ** (n_samples-1)}")
    comb = []
    for i, combination in enumerate(comb_gen):
        if i >= 500:
            break
        comb.append(combination)
    print(comb)

if __name__ == "__main__":
    main()
