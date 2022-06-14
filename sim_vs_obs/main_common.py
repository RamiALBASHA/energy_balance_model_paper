from pathlib import Path
from string import ascii_lowercase

from matplotlib import pyplot, ticker, rcParams
from numpy import full, arange
from pandas import read_csv

from sim_vs_obs.common import CMAP, NORM_INCIDENT_PAR
from utils import stats
from utils.config import UNITS_MAP

rcParams['text.usetex'] = True

EXPERIMENTS = {
    'maricopa_face': ('Maricopa FACE', 'bigleaf'),
    'maricopa_hsc': ('Maricopa HSC', 'bigleaf'),
    'grignon': ('Grignon', 'layered'),
    'braunschweig_face': ('Braunschweig FACE', 'bigleaf')}


def add_1_1_line(ax: pyplot.Subplot, lims: dict, **kwargs):
    lims = [lims[s] for s in ('min', 'max')]
    ax.plot(lims, lims, 'k--', **kwargs)
    pass


def plot_sim_vs_obs(path_source: Path, path_outputs: Path, is_corrected: bool = True, is_lumped_leaves: bool = True):
    s_corrected = 'corrected' if is_corrected else 'neutral'
    s_lumped = 'lumped' if is_lumped_leaves else 'sunlit-shaded'

    plot_kwargs = dict(marker='.', edgecolor='none', alpha=0.5, cmap=CMAP, norm=NORM_INCIDENT_PAR)
    kwargs = dict(fontsize=8, ha='right')
    t_unit = UNITS_MAP['temperature'][1]

    vars_to_plot = ('temperature_canopy', 'delta_temperature_canopy')
    lims = {s: {'min': 0, 'max': 0} for s in vars_to_plot}

    fig, axs = pyplot.subplots(nrows=2, ncols=len(EXPERIMENTS), figsize=(19 / 2.54, 12 / 2.54),
                               sharex='row', sharey='row', gridspec_kw=dict(wspace=0, hspace=0.5))
    for j, experiment in enumerate(EXPERIMENTS.keys()):
        dir_name = '_'.join((EXPERIMENTS[experiment][1], s_lumped))
        df = read_csv(path_source / experiment / 'outputs' / s_corrected / dir_name / 'results.csv')
        axs[0, j].set_title(EXPERIMENTS[experiment][0], fontsize=10, pad=20, fontweight='bold')
        for i, var in enumerate(vars_to_plot):
            ax = axs[i, j]
            for period in ('day', 'night'):
                df_ = df[df['incident_par'] > 0] if period == 'day' else df[df['incident_par'] == 0]
                sim_ = df_[f'{var}_sim']
                obs_ = df_[f'{var}_obs']
                par_ = df_['incident_par']
                ax.scatter(obs_, sim_, c=par_, label=period, **plot_kwargs)

            sim = df[f'{var}_sim']
            obs = df[f'{var}_obs']
            ax.text(0.05, 0.875, f"({ascii_lowercase[j + i * 4]})", transform=ax.transAxes)
            ax.text(0.95, 0.25, f"R² = {stats.calc_r2(sim, obs):.2f}", transform=ax.transAxes, **kwargs)
            ax.text(0.95, 0.15, f"RMSE = {stats.calc_rmse(sim, obs):.2f}", transform=ax.transAxes, **kwargs)
            ax.text(0.95, 0.05, f"nNSE = {stats.calc_nash_sutcliffe(sim, obs):.2f}", transform=ax.transAxes, **kwargs)
            lims[var]['min'] = min(lims[var]['min'], min(sim.min(), obs.min()))
            lims[var]['max'] = max(lims[var]['max'], max(sim.max(), obs.max()))
            ax.set_aspect(aspect='equal')

    for i, s, var in zip(range(len(vars_to_plot)), ('', ' depression'), vars_to_plot):
        axs[i, 0].set_ylabel(f"Simulated canopy\ntemperature{s}\n{t_unit}")
        axs[i, 1].set_xlabel(f'Measured canopy temperature{s} {t_unit}')
        axs[i, 1].xaxis.set_label_coords(1, -0.21, transform=axs[i, 1].transAxes)

    for ax_t, ax_dt, experiment in zip(axs[0, :], axs[1, :], EXPERIMENTS.values()):
        add_1_1_line(ax_t, lims=lims['temperature_canopy'], linewidth=0.5)
        add_1_1_line(ax_dt, lims=lims['delta_temperature_canopy'], linewidth=0.5)
        ax_t.tick_params(axis='both', which='major', labelsize=8)
        ax_dt.tick_params(axis='both', which='major', labelsize=8)

    axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axs[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(5))

    axs[0, 0].legend(fontsize=8, framealpha=0, loc=(0.05, 0.6), handlelength=0.25)

    fig.subplots_adjust(left=0.15, right=0.985, top=0.89, bottom=0.125)

    fig.savefig(path_outputs / f'sim_vs_obs_all_experiments_{s_corrected}_{s_lumped}.png')
    pyplot.close('all')
    pass


def plot_correction_effect(path_source: Path, path_outputs: Path):
    leaf_classes = ('lumped', 'sunlit-shaded')
    fig, axs = pyplot.subplots(nrows=len(leaf_classes), figsize=(19 / 2.54, 10 / 2.54),
                               gridspec_kw=dict(hspace=0), sharex='all', sharey='all')
    width = 0.3
    d_width = width / 1.5
    kwargs = dict(width=width, edgecolor='grey', linewidth=0.5)
    x = range(len(EXPERIMENTS))

    x_ticks = None
    x_tick_labels = None

    for ax, leaf_class in zip(axs, leaf_classes):
        x_ticks = []
        x_tick_labels = []
        for i, experiment in enumerate(EXPERIMENTS.keys()):
            for sub_dir in ('neutral', 'corrected'):
                dx = - d_width if sub_dir == 'neutral' else + d_width
                df = read_csv(path_source / ('/'.join(
                    [experiment, 'outputs', sub_dir, f'{EXPERIMENTS[experiment][1]}_{leaf_class}', 'results.csv'])))
                squared_bias, nonunity_slope, lack_of_correlation = stats.calc_mean_squared_deviation_components(
                    sim=df['delta_temperature_canopy_sim'],
                    obs=df['delta_temperature_canopy_obs'])

                x_pos = x[i] + dx
                x_ticks.append(x_pos)
                x_tick_labels.append(sub_dir)
                ax.bar(x_pos, squared_bias, label='Squared bias', color='red', **kwargs)
                ax.bar(x_pos, nonunity_slope, label='Nonunity slope', bottom=squared_bias, color='white', **kwargs)
                ax.bar(x_pos, lack_of_correlation, label='Lack of correlation', bottom=squared_bias + nonunity_slope,
                       color='blue', **kwargs)

    handles, labels_ = axs[0].get_legend_handles_labels()
    labels = sorted(set(labels_))
    handles = [handles[labels_.index(s)] for s in labels]
    axs[0].legend(handles=handles, labels=labels, framealpha=0.5, fancybox=True)
    axs[0].set_ylabel(r'Mean squared deviation ($\rm {^\circ\/C}^2$)')
    axs[0].yaxis.set_label_coords(-0.065, 0, transform=axs[0].transAxes)
    axs[0].set_ylim(0, 14)
    axs[0].set_yticks(range(0, int(max(axs[0].get_ylim())), 2))

    axs[-1].set_xticks(x_ticks, minor=True)
    axs[-1].set_xticklabels(x_tick_labels, minor=True, fontsize=8)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels([v[0] for v in EXPERIMENTS.values()])
    axs[-1].tick_params(axis='x', which='major', pad=20, length=0)
    axs[-1].tick_params(axis='x', which='minor', pad=5, length=0)
    axs[-1].set_xlim(-0.55, 3.5)

    for ax, s in zip(axs, ascii_lowercase):
        ax.text(0.01, 0.9, f'({s})', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path_outputs / 'correction_effect.png')
    pyplot.close('all')
    pass


def plot_correction_effect2(path_source: Path, leaf_class: str, path_outputs: Path):
    pyplot.close()
    fig, axs = pyplot.subplots(nrows=3, figsize=(14 / 2.54, 15 / 2.54), gridspec_kw=dict(hspace=0), sharex='col')
    width = 0.2
    d_width = width / 1.5
    kwargs = dict(width=width, edgecolor='grey', linewidth=0.5)
    x = range(len(EXPERIMENTS))

    x_ticks = []
    x_tick_labels = []
    for i, experiment in enumerate(EXPERIMENTS.keys()):
        for sub_dir in ('neutral', 'corrected'):
            df = read_csv(path_source / ('/'.join(
                [experiment, 'outputs', sub_dir, f'{EXPERIMENTS[experiment][1]}_{leaf_class}', 'results.csv'])))
            squared_bias, nonunity_slope, lack_of_correlation = stats.calc_mean_squared_deviation_components(
                sim=df['delta_temperature_canopy_sim'],
                obs=df['delta_temperature_canopy_obs'])

            dx = - d_width if sub_dir == 'neutral' else + d_width
            x_pos = x[i] + dx
            x_ticks.append(x_pos)
            x_tick_labels.append(sub_dir)
            axs[-1].bar(x_pos, squared_bias, label='Squared bias', color='red', **kwargs)
            axs[-1].bar(x_pos, nonunity_slope, label='Nonunity slope', bottom=squared_bias, color='white', **kwargs)
            axs[-1].bar(x_pos, lack_of_correlation, label='Lack of correlation', bottom=squared_bias + nonunity_slope,
                        color='blue', **kwargs)
            if sub_dir == 'neutral':
                axs[1].boxplot(df['delta_temperature_canopy_obs'], positions=[i], showfliers=False, widths=[width * 2])

    for i, experiment in enumerate(EXPERIMENTS.keys()):
        df_cart = read_csv(path_source / (
            f"{experiment}/outputs/neutral/{EXPERIMENTS[experiment][1]}_{leaf_class}/results_cart.csv"))
        axs[0].boxplot(df_cart['wind_speed'], positions=[i], showfliers=False, widths=[width * 2])

    handles, labels_ = axs[-1].get_legend_handles_labels()
    labels = sorted(set(labels_))
    handles = [handles[labels_.index(s)] for s in labels]
    axs[-1].legend(handles=handles, labels=labels, framealpha=0.5, fancybox=True, fontsize=8)

    axs[0].set_ylabel('\n'.join(['Observed\nwind speed', r'($\rm m\/s^{-1}$)']))
    axs[1].set_ylabel('\n'.join(['Observed canopy\ntempreature depression', r'($\rm ^\circ\/C$)']))
    # axs[0].yaxis.set_label_coords(-0.065, 0, transform=axs[0].transAxes)
    axs[-1].set_ylim(0, 14)
    # axs[-1].set_yticks(range(0, int(max(axs[-1].get_ylim())), 2))

    axs[-1].set_xticks(x_ticks, minor=True)
    axs[-1].set_xticklabels(x_tick_labels, minor=True, fontsize=9, rotation=90)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels([v[0].replace(' ', '\n') for v in EXPERIMENTS.values()], rotation=0, fontsize=11)
    axs[-1].tick_params(axis='x', which='major', pad=55, length=0)
    axs[-1].tick_params(axis='x', which='minor', pad=5, length=0)
    axs[-1].set_xlim(-0.55, 3.5)
    axs[-1].set_ylabel('\n'.join(['Mean squared\ndeviation', r'($\rm {^\circ\/C}^2$)']))

    for ax, s in zip(axs, ascii_lowercase):
        ax.text(0.01, 0.875, f'({s})', transform=ax.transAxes)

    # fig.subplots_adjust(left=0.2, right=0.99, bottom=0.25)
    fig.tight_layout()
    fig.savefig(path_outputs / f'correction_effect_{leaf_class}.png')
    pyplot.close('all')
    pass


def plot_correction_effect3(path_source: Path, leaf_class: str, path_outputs: Path):
    pyplot.close()
    fig, axs = pyplot.subplots(ncols=4, figsize=(19 / 2.54, 8 / 2.54),
                               gridspec_kw=dict(wspace=0, width_ratios=[5, 2, 2, 2]), sharey='row')

    [ax.clear() for ax in axs]
    height = 0.2
    d_height = height / 1.5
    kwargs = dict(height=height, edgecolor='grey', linewidth=0.5)
    y = range(len(EXPERIMENTS))

    y_ticks = []
    y_tick_labels = []
    for i, experiment in enumerate(EXPERIMENTS.keys()):
        for sub_dir in ('neutral', 'corrected'):
            df = read_csv(path_source / ('/'.join(
                [experiment, 'outputs', sub_dir, f'{EXPERIMENTS[experiment][1]}_{leaf_class}', 'results.csv'])))
            squared_bias, nonunity_slope, lack_of_correlation = stats.calc_mean_squared_deviation_components(
                sim=df['delta_temperature_canopy_sim'],
                obs=df['delta_temperature_canopy_obs'])

            dy = - d_height if sub_dir == 'neutral' else + d_height
            y_pos = y[i] + dy
            y_ticks.append(y_pos)
            y_tick_labels.append(sub_dir)
            axs[0].barh(y_pos, squared_bias, label='Squared bias', color='blue', **kwargs)
            axs[0].barh(y_pos, nonunity_slope, label='Nonunity slope', left=squared_bias, color='white', **kwargs)
            axs[0].barh(y_pos, lack_of_correlation, label='Lack of correlation', left=squared_bias + nonunity_slope,
                        color='red', **kwargs)

    for i, experiment in enumerate(EXPERIMENTS.keys()):
        df_cart = read_csv(path_source / (
            f"{experiment}/outputs/neutral/{EXPERIMENTS[experiment][1]}_{leaf_class}/results_cart.csv"))
        axs[1].boxplot(df_cart['temperature_air'], positions=[i], showfliers=False, vert=False)
        axs[2].boxplot(df_cart['temperature_canopy_obs'] - df_cart['temperature_air'], positions=[i], showfliers=False,
                       vert=False)
        axs[3].boxplot(df_cart['wind_speed'], positions=[i], showfliers=False, vert=False)

    handles, labels_ = axs[0].get_legend_handles_labels()
    labels = sorted(set(labels_))
    handles = [handles[labels_.index(s)] for s in labels]
    labels = [s if not 'Lack' in s else 'Lack of\ncorrelation' for s in labels]
    axs[0].legend(handles=handles, labels=labels, framealpha=0.5, fancybox=True, fontsize=8, loc='lower right',
                  handlelength=1)

    axs[0].set(xlim=(0, 14), ylim=(-0.75, 3.5))
    axs[0].set_yticks(y_ticks, minor=True)
    axs[0].set_yticklabels(y_tick_labels, minor=True, fontsize=9)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([v[0] for v in EXPERIMENTS.values()], ha='left')
    axs[0].tick_params(axis='y', which='major', pad=150, length=0)
    axs[0].tick_params(axis='y', which='minor', pad=5, length=0)
    axs[0].set_xlabel('\n'.join(['Mean squared\ndeviation', r'($\rm {^\circ\/C}^2$)']))

    axs[1].set_xlabel('\n'.join(['Air\ntempreature', r'($\rm ^\circ\/C$)']))
    axs[1].yaxis.set_visible(False)

    axs[2].vlines(0, *axs[1].get_ylim(), color='grey', linestyles='--', linewidth=0.5, zorder=0)
    axs[2].set_xlabel('\n'.join(['Canopy\ntempreature\ndepression', r'($\rm ^\circ\/C$)']))
    axs[2].set_xlim(-13, 13)
    axs[2].yaxis.set_visible(False)

    axs[3].set(xlim=(-1, 7), xlabel='\n'.join(['Wind\nspeed', r'($\rm m\/s^{-1}$)']))
    axs[3].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[3].yaxis.set_visible(False)

    for ax, s in zip(axs, ascii_lowercase):
        ax.tick_params(axis='x', labelsize=9)
        ax.text(0.05, 0.925, f'({s})', transform=ax.transAxes)

    axs[0].invert_yaxis()

    fig.tight_layout()
    fig.savefig(path_outputs / f'correction_effect_{leaf_class}_2.png')
    pyplot.close('all')
    pass


def plot_stability_vs_leaf_category_heatmap(path_source: Path, path_outputs: Path):
    stability_correction_cases = ('neutral', 'corrected')
    leaf_classes = ('lumped', 'sunlit-shaded')
    experiments = list(EXPERIMENTS.keys())
    data = full(shape=(len(experiments) * len(leaf_classes), len(stability_correction_cases)),
                fill_value=None).astype(float)
    for i_experiment, experiment in enumerate(experiments):
        dy = i_experiment * len(leaf_classes)
        for i_leaf_class, leaf_category in enumerate(leaf_classes):
            i = i_leaf_class + dy
            for j, stability_correction in enumerate(stability_correction_cases):
                subdir = f"{experiment}/outputs/{stability_correction}/{EXPERIMENTS[experiment][1]}_{leaf_category}"
                df = read_csv(path_source / subdir / 'results.csv')
                data[i, j] = stats.calc_mean_squared_deviation(
                    sim=df['delta_temperature_canopy_sim'],
                    obs=df['delta_temperature_canopy_obs'])

    fig, ax = pyplot.subplots(figsize=(9 / 2.54, 9 / 2.54))
    im = ax.imshow(data, cmap='Oranges', aspect='auto')
    ax.set(xticks=arange(data.shape[1]), xticklabels=stability_correction_cases,
           yticks=arange(data.shape[0]), yticklabels=leaf_classes * len(experiments))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=8)
    ax.tick_params(axis='x', rotation=90)

    for i_experiment, experiment in enumerate(experiments):
        if experiment != experiments[-1]:
            ax.hlines(i_experiment * len(leaf_classes) + 1.5, *ax.get_xlim(), color='k', linewidth=1)
        ax.text(-7.75, i_experiment * len(leaf_classes) + 0.5,
                f'({ascii_lowercase[i_experiment]}) {EXPERIMENTS[experiment][0]}')

    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    cbar.ax.set_ylabel(r'Mean squared deviation ($\rm {^\circ\/C}^2$)', ha="right", va='center', rotation=0)

    fig.subplots_adjust(left=0.75, right=0.95, bottom=0.05, top=0.8)
    fig.savefig(path_outputs / 'correction_and_leaf_class_effects.png')
    pyplot.close('all')
    pass


def plot_per_richardson_zone(path_source: Path, leaf_class: str, path_outputs: Path):
    ri_up_thresholds = {
        'unstable': -0.01,
        'neutral': 0.01,
        'stable': 0.2
    }

    stability_zones = list(ri_up_thresholds.keys())
    experiment_names = list(EXPERIMENTS.keys())

    count_dict = {k1: {k2: None} for k2 in EXPERIMENTS.keys() for k1 in stability_zones}
    error_dict = {k1: {k2: None} for k2 in EXPERIMENTS.keys() for k1 in stability_zones}

    for k, v in EXPERIMENTS.items():
        df_corrected = read_csv(path_source / f"{k}/outputs/corrected/{v[1]}_{leaf_class}/results_cart.csv")
        # df_neutral = read_csv(path_source / f"{k}/outputs/neutral/{v[1]}_sunlit-shaded/results_cart.csv")
        # ri = df_neutral.apply(lambda x: calc_monin_obukhov_obs(
        #     friction_velocity=x['friction_velocity'],
        #     temperature_canopy=x['temperature_canopy_obs'],
        #     temperature_air=x['temperature_air'],
        #     aerodynamic_resistance=x['neutral_aerodynamic_resistance']), axis=1)
        ri = df_corrected['richardson']

        ri_idx = {}
        richardson_threshold_labels = {}
        for i_stability, stability_zone in enumerate(stability_zones):
            if i_stability == 0:
                ri_idx[stability_zone] = ri <= ri_up_thresholds[stability_zone]
                richardson_threshold_labels[stability_zone] = ' '.join(
                    (r'Ri $\rm \leq$', str(ri_up_thresholds[stability_zone])))
            else:
                ri_idx[stability_zone] = ((ri > ri_up_thresholds[stability_zones[i_stability - 1]]) & (
                        ri <= ri_up_thresholds[stability_zone]))
                richardson_threshold_labels[stability_zone] = ' '.join(
                    (str(ri_up_thresholds[stability_zones[i_stability - 1]]),
                     r'< Ri $\rm \leq$',
                     str(ri_up_thresholds[stability_zone])))

        t_sim_corrected = df_corrected['temperature_canopy_sim']
        t_obs = df_corrected['temperature_canopy_obs']

        df_neutral = read_csv(path_source / f"{k}/outputs/neutral/{v[1]}_{leaf_class}/results_cart.csv")
        t_sim_neutral = df_neutral['temperature_canopy_sim']

        for k_ri, v_ri in ri_idx.items():
            print(k_ri, v_ri[v_ri].count())
            count_dict[k_ri][k] = v_ri[v_ri].count() / ri.count() * 100
            error_abs_neutral = (t_sim_neutral[v_ri] - t_obs[v_ri]).apply(lambda x: abs(x)).mean()
            error_abs_corrected = (t_sim_corrected[v_ri] - t_obs[v_ri]).apply(lambda x: abs(x)).mean()
            error_dict[k_ri][k] = error_abs_neutral - error_abs_corrected

    pyplot.close()
    fig, (ax_count, ax_error) = pyplot.subplots(ncols=2, sharey='all', gridspec_kw=dict(wspace=0))

    y_ticks = []
    y_tick_labels = []
    color_seq = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    for i_stability, stability_zone in enumerate(stability_zones):
        y_shift = i_stability * (len(experiment_names) + 1)
        for i_experiment, experiment_name in enumerate(experiment_names):
            y_position = i_experiment + y_shift
            ax_count.barh(y_position, count_dict[stability_zone][experiment_name], facecolor=color_seq[i_experiment])
            ax_error.barh(y_position, error_dict[stability_zone][experiment_name], facecolor=color_seq[i_experiment])
            y_ticks.append(y_position)
            y_tick_labels.append(EXPERIMENTS[experiment_name][0])
        pass
    pass

    ax_count.set(xlabel="Percentage of\nsimulated time steps",
                 ylim=(-1.5, 14))
    ax_count.set_yticks(
        [i * (len(experiment_names) + 1) + len(experiment_names) / 2.5 for i in range(len(stability_zones))])
    ax_count.set_yticklabels([s.capitalize() for s in stability_zones], ha='left')

    ax_count.set_yticks(y_ticks, minor=True)
    ax_count.set_yticklabels(y_tick_labels, minor=True, fontsize=8)

    ax_count.tick_params(axis='y', which='major', pad=125, length=0)
    ax_count.tick_params(axis='y', which='minor', pad=5, length=0, rotation=0)

    ax_error.set_xlabel('\n'.join(['Reduction in mean', 'absolute temperature error', r'($\rm ^\circ\/C$)']))

    for ax, s in zip((ax_count, ax_error), ascii_lowercase):
        ax.text(0.025, 0.95, f'({s})', transform=ax.transAxes)

    ax_count.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path_outputs / 'result_per_richardson_zone.png')


if __name__ == '__main__':
    path_sources = Path(__file__).parents[1] / 'sources'
    path_fig = path_sources / 'figs'
    plot_correction_effect(path_source=path_sources, path_outputs=path_fig)
    plot_stability_vs_leaf_category_heatmap(path_source=path_sources, path_outputs=path_fig)

    for is_lumped in (True, False):
        plot_sim_vs_obs(
            path_source=path_sources,
            path_outputs=path_fig,
            is_corrected=True,
            is_lumped_leaves=is_lumped)
        plot_correction_effect2(
            path_source=path_sources,
            leaf_class='lumped' if is_lumped else 'sunlit-shaded',
            path_outputs=path_fig)
        plot_correction_effect3(
            path_source=path_sources,
            leaf_class='lumped' if is_lumped else 'sunlit-shaded',
            path_outputs=path_fig)
        plot_per_richardson_zone(
            path_source=path_sources,
            leaf_class='lumped' if is_lumped else 'sunlit-shaded',
            path_outputs=path_fig)
