from datetime import datetime
from pathlib import Path

from matplotlib import pyplot, ticker
from pandas import read_csv


def plot_obs(path_phyllo_temperature: Path, path_gai: Path, path_fig: Path):
    df_t_leaf = read_csv(path_phyllo_temperature, sep=';', decimal='.', comment='#')
    df_t_leaf.loc[:, 'time'] = df_t_leaf.apply(lambda x: datetime.strptime(x['time'], '%Y-%m-%d %H:%M'), axis=1)

    df_gai = read_csv(path_gai, sep=';', decimal='.', comment='#')
    df_gai.loc[:, 'date'] = df_gai.apply(lambda x: datetime.strptime(x['date'], '%Y-%m-%d').date(), axis=1)
    df_gai.set_index('date', inplace=True)
    treatments = df_t_leaf['treatment'].unique()

    counter = 0
    for date_obs, row in df_gai.iterrows():
        gai_cover = row['avg']
        for hour in df_t_leaf[df_t_leaf['time'].dt.date == date_obs]['time'].unique():
            counter += 1
            fig, axs = pyplot.subplots(nrows=2, sharex='all', sharey='all')

            for i, treatment in enumerate(treatments):
                ax = axs[i]
                ax.set_title(treatment)
                df1 = df_t_leaf[(df_t_leaf['time'].dt.date == date_obs) & (df_t_leaf['treatment'] == treatment)]
                leaf_layers = df1['leaf_level'].unique()
                for leaf_layer in leaf_layers:
                    t = df1[(df1['leaf_level'] == leaf_layer) & (df1['time'] == hour)]['temperature']
                    ax.scatter(t, [leaf_layer] * len(t), marker='s', c='red', alpha=0.3)

            axs[0].set(ylim=(0, 13), xlim=(-5, 25))
            [ax.set_ylabel('layer index') for ax in axs]
            axs[1].set_xlabel(r'$\mathregular{T_{leaf}\/[^\circ C]}$')
            axs[0].yaxis.set_major_locator(ticker.MultipleLocator(1))

            fig.suptitle(f"{str(hour).replace('T', ' ')[:-10]} (%GAI={gai_cover:.2f})")
            fig.savefig(path_fig / f'{counter}.png')
            pyplot.close()

    pass


if __name__ == "__main__":
    path_root = Path(__file__).parent
    path_figs = path_root / 'figs'
    path_figs.mkdir(parents=True, exist_ok=True)
    plot_obs(
        path_phyllo_temperature=path_root / 'temperatures_phylloclimate.csv',
        path_gai=path_root / 'gai_percentage.csv',
        path_fig=path_figs)
