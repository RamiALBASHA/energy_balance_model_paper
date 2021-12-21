import matplotlib.pyplot as plt
from matplotlib import cm, colors


def compare_temperature(obs: list, sim: list, ax: plt.Subplot = None, return_ax: bool = False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.scatter(obs, sim, c=range(len(obs)), cmap=cm.seismic, norm=colors.Normalize(vmin=0, vmax=len(obs)))
    ax.set(xlabel='obs', ylabel='sim')
    ax_lims = sorted([v for sub_list in (ax.get_xlim(), ax.get_ylim()) for v in sub_list])
    xylims = [ax_lims[i] for i in (0, -1)]
    ax.set(xlim=xylims, ylim=xylims)
    ax.plot(xylims, xylims, 'k--')
    if return_ax:
        return ax
