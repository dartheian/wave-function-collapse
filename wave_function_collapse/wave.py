from abc import ABCMeta, abstractmethod
from numpy import full, ones, where
from scipy.stats import entropy


class Wave(metaclass=ABCMeta):
    def __init__(self, shape, counts):
        self.possible = ones((counts.size, *shape), dtype=int, order='F')
        self.counts = counts

    @property
    def field_shape(self):
        return self.possible.shape[1:]

    @property
    def state_set_cardinality(self):
        return self.counts.size

    @property
    def histogram(self):
        return self.possible * self.counts

    @property
    def density(self):
        return self.histogram / self.histogram.sum(axis=-1, keepdims=True)

    @property
    def entropy(self):
        return entropy(self.density, base=self.state_set_cardinality, axis=-1)

    @property
    def min_entropy(self):
        min_nonzero = self.entropy[self.entropy != 0].min()
        return where(self.entropy == min_nonzero)

    @property
    def collapsed(self):
        return self.density == 1

    @property
    def view(self):
        collapsed = where(self.collapsed)
        view = full(self.field_shape, -1)
        view[collapsed[:-1]] = collapsed[-1]
        return view

    def collapse(self, position, state):
        self.possible[*position] = 0
        self.possible[*position, state] = 1

    def observe(self, position, rng):
        state = rng.choice(
            self.state_set_cardinality, p=self.density[position])
        self.fix(position, state)

    def fix(self, position, state):
        self.collapse(position, state)
        collapsed = self.collapsed
        self.propagate_information(position, state)
        while (collapsed != self.collapsed).any():
            position = where(collapsed != self.collapsed)
            collapsed = self.collapsed
            self.propagate_information(position[:-1], position[-1])

    @abstractmethod
    def propagate_information(self, position, state):
        raise NotImplementedError
