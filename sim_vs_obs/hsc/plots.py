import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator

from utils import stats


def compare_temperature(obs: list, sim: list, ax: plt.Subplot = None, return_ax: bool = False,
                        plot_colorbar: bool = True, write_stats: bool = False):
    if ax is None:
        fig, ax = plt.subplots()
    cmap = cm.seismic
    norm = colors.Normalize(vmin=0, vmax=len(obs))
    ax.scatter(obs, sim, c=range(len(obs)), cmap=cmap, norm=norm)
    if plot_colorbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal")
        cbar.ax.set_xlabel('hour', va="bottom")
        cbar.ax.xaxis.set_major_locator(MultipleLocator(4))

    ax.set(xlabel='obs', ylabel='sim')
    ax_lims = sorted([v for sub_list in (ax.get_xlim(), ax.get_ylim()) for v in sub_list])
    xylims = [ax_lims[i] for i in (0, -1)]
    ax.set(xlim=xylims, ylim=xylims)
    ax.plot(xylims, xylims, 'k--')

    if write_stats:
        ax.text(0.1, 0.9, f"R2 = {stats.calc_r2(sim, obs):.3f}", transform=ax.transAxes)
        ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(sim, obs):.3f}", transform=ax.transAxes)
    if return_ax:
        return ax
