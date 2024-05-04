#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import math

rng = numpy.random.default_rng()


def observations():
    sudoku = numpy.loadtxt("data.csv", dtype=int, delimiter=",")
    clues_indices = sudoku.nonzero()
    clues_values = sudoku[clues_indices] - 1
    return (clues_indices, clues_values)


def initialize_wave(dim, state_space_size):
    view = (*dim, state_space_size)
    p = 1 / state_space_size
    return numpy.full(view, p)


def observe(wave, position):
    p = wave[*position]
    state = rng.choice(p.size, p=p)
    collapse(wave, position, state)
    propagate_information(wave, position, state)


def collapse(wave, position, state):
    wave[*position] = 0.0
    wave[*position, state] = 1.0


def propagate_information(wave, position, state):
    # Row and column
    wave[:, position[1], state] = 0.0
    wave[position[0], :, state] = 0.0
    # Neighbors
    f0 = position[0] // 3 * 3
    c0 = position[0] // 3 * 3 + 3
    f1 = position[1] // 3 * 3
    c1 = position[1] // 3 * 3 + 3
    wave[f0:c0, f1:c1, state] = 0.0
    # Reset state
    wave[*position, state] = 1.0


def compute_entropy(wave):
    base_change_factor = math.log(math.e, 9)
    self_information = - base_change_factor * numpy.log(wave, where=wave != 0)
    return numpy.sum(wave * self_information, axis=-1)


def inspect(wave):
    view = numpy.sum(wave, where=wave == 1, axis=2, dtype=int)
    i = numpy.where(wave == 1.0)
    view[view == 1] = i[-1]
    return view


sudoku = numpy.loadtxt("data.csv", dtype=int, delimiter=",")
wave = initialize_wave((9, 9), 9)
(position, state) = observations()
collapse(wave, position, state)
for p in zip(*position):
    observe(wave, p)
entropy = compute_entropy(wave)
view = inspect(wave)
