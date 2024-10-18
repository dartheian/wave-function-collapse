from numpy import asfortranarray
from numpy.random import default_rng
from wave_function_collapse.sudoku_wave import SudokuWave

# path = "data/hard.csv"

# def run(path, seed):
#     rng = default_rng(seed)
#     wave = SudokuWave.from_file(path, rng)
#     while wave.entropy.sum() != 0:
#         position = wave.min_entropy
#         i = rng.choice(position[0].size)
#         wave.observe((position[0][i], position[1][i]))
#     return wave


# rng = default_rng(0)
# wave = SudokuWave.from_file(path, rng)

# h = wave.histogram
# s = wave.histogram.sum(axis=-1)

# def cmp(a, b):
#     print(a / b == wave.density)

wave = SudokuWave([], [], None)
