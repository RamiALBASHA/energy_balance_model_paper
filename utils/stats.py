from typing import Union

from numpy import corrcoef, array

Vector = Union[list, tuple, array]


def calc_r2(vector_1: Vector, vector_2: Vector) -> float:
    return corrcoef(array(vector_1), array(vector_2))[0, 1]


def calc_rmse(vector_1: Vector, vector_2: Vector) -> float:
    return (sum([(x - y) ** 2 for x, y in zip(vector_1, vector_2)]) / len(vector_1)) ** 0.5


def calc_mean(vector: Vector):
    return sum(vector) / len(vector)


def calc_nash_sutcliffe(sim: Vector, obs: Vector) -> float:
    obs_mean = calc_mean(vector=obs)
    sim_variance = []
    obs_variance = []
    for sim_value, obs_value in zip(sim, obs):
        sim_variance.append((sim_value - obs_value) ** 2)
        obs_variance.append((obs_value - obs_mean) ** 2)

    return 1. - sum(sim_variance) / sum(obs_variance)


def calc_normaized_nash_sutcliffe(sim: Vector, obs: Vector) -> float:
    return 1. / (2. - calc_nash_sutcliffe(sim=sim, obs=obs))
