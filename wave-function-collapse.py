#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import entropy
from numpy import full, loadtxt, ones, where, stack
from numpy.random import default_rng


class Wave:
    def __init__(self, shape, counts, rng):
        self.possible = ones((*shape, counts.size), dtype=int)
        self.counts = counts
        self.rng = rng

    @property
    def field_shape(self):
        return self.possible.shape[:-1]

    @property
    def states_number(self):
        return self.counts.size

    @property
    def histogram(self):
        return self.possible * self.counts

    @property
    def density(self):
        return self.histogram / self.histogram.sum(axis=-1, keepdims=True)

    @property
    def entropy(self):
        return entropy(self.density, base=self.states_number, axis=-1)

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

    def observe(self, position):
        state = self.rng.choice(self.states_number, p=self.density[position])
        self.fix(position, state)

    def fix(self, position, state):
        self.collapse(position, state)
        collapsed = self.collapsed
        self.propagate_information(position, state)
        while (collapsed != self.collapsed).any():
            position = where(collapsed != self.collapsed)
            collapsed = self.collapsed
            self.propagate_information(position[:-1], position[-1])

    def propagate_information(self, position, state):
        raise NotImplementedError


v = []
e = []
d = []


class SudokuWave(Wave):
    def __init__(self, fixed_positions, fixed_states, rng):
        super().__init__((9, 9), ones(9), rng)
        self.fix(fixed_positions, fixed_states)

    def propagate_information(self, position, state):
        self.possible[position[0], :, state] = 0
        self.possible[:, position[1], state] = 0
        for x_offset in range(3):
            for y_offset in range(3):
                x = position[0] // 3 * 3 + x_offset
                y = position[1] // 3 * 3 + y_offset
                self.possible[x, y, state] = 0
        self.possible[*position, state] = 1
        v.append(self.view)
        e.append(self.entropy)
        d.append(self.density)

    @property
    def view(self):
        return super().view + 1

    @staticmethod
    def from_file(file, seed):
        sudoku = loadtxt(file, dtype=int, delimiter=",")
        clues_positions = sudoku.nonzero()
        clues_values = sudoku[clues_positions] - 1
        return SudokuWave(clues_positions, clues_values, seed)

    @property
    def valid(self):
        rule = [45, 45, 45, 45, 45, 45, 45, 45, 45]
        a = (self.view.sum(axis=0) == rule).all()
        b = (self.view.sum(axis=1) == rule).all()
        c = True
        for x in range(3):
            for y in range(3):
                if self.view[x*3:x*3+3, y*3:y*3+3].sum() != 45:
                    c = False
        return a and b and c


def run(seed):
    rng = default_rng(seed)
    wave = SudokuWave.from_file("data.csv", rng)
    v.append(wave.view)
    e.append(wave.entropy)
    d.append(wave.density)
    while wave.entropy.sum() != 0:
        position = wave.min_entropy
        i = rng.choice(position[0].size)
        wave.observe((position[0][i], position[1][i]))
        v.append(wave.view)
        e.append(wave.entropy)
        d.append(wave.density)
    return wave


wave = run(1)
v = stack(v)
e = stack(e)
d = stack(d)
