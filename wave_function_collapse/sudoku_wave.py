from numpy import loadtxt, ones
from .wave import Wave


class SudokuWave(Wave):
    def __init__(self, fixed_positions, fixed_states):
        super().__init__(ones(9), (9, 9))
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
                if self.view[x * 3:x * 3 + 3, y * 3:y * 3 + 3].sum() != 45:
                    c = False
        return a and b and c
