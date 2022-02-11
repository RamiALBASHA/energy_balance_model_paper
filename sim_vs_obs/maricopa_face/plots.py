from matplotlib import pyplot
from pandas import DataFrame

from sim_vs_obs.maricopa_face import base_functions
from sim_vs_obs.maricopa_face.config import PathInfos


def plot_irradiance(shoot_obj: dict, obs_df: DataFrame):
    fig, axs = pyplot.subplots(nrows=2, ncols=2)

    obs_faparc, sim_faparc = zip(*[(obs_df.loc[idx, 'fAPARc'], base_functions.calc_sim_fapar(shoot_obj[idx]))
                                   for idx in obs_df.index])

    axs[0, 0].scatter(obs_faparc, sim_faparc, marker='.', alpha=0.2, label='fAPARc')
    axs[0, 0].plot([0, 1], [0, 1], linestyle='--', color='grey', label='1:1')
    axs[0, 0].set(xlabel='obs', ylabel='sim')

    for ax in (axs[1, 0], axs[0, 1]):
        ax.scatter(obs_df['gai'], obs_df['fAPARc'], marker='.', alpha=0.2, color='r', label='obs')

    for ax in (axs[1, 0], axs[1, 1]):
        ax.scatter([sum(shoot.inputs.leaf_layers.values()) for shoot in shoot_obj.values()], sim_faparc,
                   marker='.', alpha=0.2, color='b', label='sim')

    for ax in (axs[1, 0], axs[0, 1], axs[1, 1]):
        ax.set(ylabel='fAPARc', xlabel='GAI')

    for ax in axs.flatten():
        ax.legend()

    fig.savefig(PathInfos.source_figs.value / 'irradiance.png')
    pyplot.close(fig)
    pass
