from typing import Union

from numpy import corrcoef, array

Vector = Union[list, tuple, array]


def calc_r2(vector_1: Vector, vector_2: Vector) -> float:
    return corrcoef(array(vector_1), array(vector_2))[0, 1]


def calc_rmse(vector_1: Vector, vector_2: Vector) -> float:
    return (sum([(x - y) ** 2 for x, y in zip(vector_1, vector_2)]) / len(vector_1)) ** 0.5


def calc_mean(vector: Vector):
    return sum(vector) / len(vector)
