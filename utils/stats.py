from typing import Union

from numpy import corrcoef, array

Vector = Union[list, tuple, array]


def calc_r2(vector_1: Vector, vector_2: Vector) -> float:
    return corrcoef(array(vector_1), array(vector_2))[0, 1] ** 2


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


def calc_mean_squared_deviation_components(sim: Vector, obs: Vector) -> tuple[float, float, float]:
    sb = calc_squared_bias(sim=sim, obs=obs)
    nu = calc_nonunity_slope(sim=sim, obs=obs)
    lc = calc_lack_of_correlation(sim=sim, obs=obs)
    return sb, nu, lc


def calc_mean_squared_deviation(sim: Vector, obs: Vector) -> float:
    return sum([(v_sim - v_obs) ** 2 for v_sim, v_obs in zip(sim, obs)]) / len(sim)


def calc_squared_bias(sim: Vector, obs: Vector) -> float:
    return (calc_mean(sim) - calc_mean(obs)) ** 2


def calc_nonunity_slope(sim: Vector, obs: Vector) -> float:
    slope = calc_slope(sim=sim, obs=obs)
    sim_mean = calc_mean(sim)
    sim_variance = sum([(v - sim_mean) ** 2 for v in sim]) / len(sim)

    return (1 - slope) ** 2 * sim_variance


def calc_lack_of_correlation(sim: Vector, obs: Vector) -> float:
    obs_mean = calc_mean(obs)
    obs_variance = sum([(v - obs_mean) ** 2 for v in obs]) / len(obs)
    r2 = calc_correlation(sim=sim, obs=obs)
    # r2 = calc_r2(sim, obs)

    return (1 - r2) * obs_variance


def calc_slope(sim: Vector, obs: Vector) -> float:
    sim_mean = calc_mean(sim)
    obs_mean = calc_mean(obs)

    numerator_ls = []
    denominator_ls = []
    for v_sim, v_obs in zip(sim, obs):
        numerator_ls.append((v_sim - sim_mean) * (v_obs - obs_mean))
        denominator_ls.append((v_sim - sim_mean) ** 2)
    return sum(numerator_ls) / sum(denominator_ls)


def calc_correlation(sim: Vector, obs: Vector) -> float:
    sim_mean = calc_mean(sim)
    obs_mean = calc_mean(obs)

    sim_deviation_ls, obs_deviation_ls = zip(*[(v_sim - sim_mean, v_obs - obs_mean) for v_sim, v_obs in zip(sim, obs)])
    covariance = sum([v_sim * v_obs for v_sim, v_obs in zip(sim_deviation_ls, obs_deviation_ls)]) ** 2
    variance_product = sum([v ** 2 for v in sim_deviation_ls]) * sum([v ** 2 for v in obs_deviation_ls])
    return covariance / variance_product
