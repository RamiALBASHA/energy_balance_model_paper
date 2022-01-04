import datetime
from datetime import timedelta

from crop_energy_balance.solver import Solver
from matplotlib import pyplot
from pandas import read_excel

from sim_vs_obs.hsc import plots
from sim_vs_obs.hsc.base_functions import get_weather_data, set_energy_balance_inputs, calc_apparent_temperature
from sim_vs_obs.hsc.config import PathInfos, WeatherInfo
from utils import stats

if __name__ == '__main__':

    fig, ax = pyplot.subplots()

    path_source_raw = PathInfos.source_raw.value
    path_source_fmt = PathInfos.source_fmt.value
    path_figs = PathInfos.source_fmt.value.parent / 'figs'
    path_figs.mkdir(parents=True, exist_ok=True)

    crop_representation = 'layered_sunlit-shaded'
    number_leaf_layers = 5

    is_bigleaf = 'bigleaf' in crop_representation
    is_lumped = 'lumped' in crop_representation
    if is_bigleaf:
        number_leaf_layers = 1

    crop_df = read_excel(path_source_raw / '3. Crop response data/Biomass_Yield_Area_Phenology.ods',
                         engine='odf', sheet_name='Time_Series_Biom_Yield_Area_N', skiprows=4)
    crop_df = crop_df[
        (crop_df['Trt'] != 'C') &
        (crop_df['Trt'] != 'H') &
        (~(crop_df['LAI'].isna())) &
        (crop_df['LAI'] > 0)
        ]

    uncertain_canopy_temperature = [
        ('R', 5, datetime.datetime(2008, 3, 18)),
        ('R', 7, datetime.datetime(2008, 3, 18)),
        ('R', 12, datetime.datetime(2008, 3, 18)),
        ('R', 5, datetime.datetime(2008, 4, 15)),
        ('R', 7, datetime.datetime(2008, 4, 15)),
        ('R', 12, datetime.datetime(2008, 4, 15)),
    ]

    for ex_trt, ex_pid, ex_date in uncertain_canopy_temperature:
        crop_df = crop_df[~((crop_df['Trt'] == ex_trt) & (crop_df['Plot'] == ex_pid) & (crop_df['date'] == ex_date))]

    weather_df = read_excel(path_source_fmt / 'Weather_Canopy_Temp_Heater_hourly_data.xlsx', sheet_name='HSC_HOURLY')

    soil_df = read_excel(path_source_fmt / 'soil_data.xlsx', sheet_name='data')
    soil_df.loc[:, 'datetime'] = soil_df.apply(lambda x: x['DATE'] + timedelta(hours=x['Time']), axis=1)

    all_sim = []
    all_obs = []
    for row_index, row in crop_df.iterrows():
        print(row_index)
        fig2, ax2 = pyplot.subplots(3, 3, sharex='col', figsize=(8, 8))

        leaf_layers = {0: row['GAI']} if is_bigleaf else {i: row['GAI'] / number_leaf_layers for i in
                                                          range(1, number_leaf_layers + 1)}
        date_obs = row['date']

        crop_weather = get_weather_data(
            raw_df=weather_df[weather_df['Date'] == date_obs],
            canopy_height=row['pl.height'] / 100.,
            treatment_id=row['Trt'],
            plot_id=row['Plot'],
            measurement_height=WeatherInfo.measurement_height.value,
            reference_height=WeatherInfo.reference_height.value,
            latitude=WeatherInfo.latitude.value,
            atmospheric_pressure=WeatherInfo.atmospheric_pressure.value)

        solvers = []
        for datetime_obs, w_data in crop_weather.iterrows():
            eb_inputs, eb_params = set_energy_balance_inputs(
                leaf_layers=leaf_layers,
                is_bigleaf=is_bigleaf,
                is_lumped=is_lumped,
                datetime_obs=datetime_obs,
                crop_data=row,
                soil_data=soil_df,
                weather_data=w_data)

            solver = Solver(leaves_category='lumped' if is_lumped else 'sunlit_shaded',
                            inputs_dict=eb_inputs,
                            params_dict=eb_params)
            solver.run(is_stability_considered=True)
            solvers.append(solver)

        temp_obs = crop_weather['canopy_temperature']
        temp_sim_source = [eb_solver.crop.state_variables.source_temperature - 273.15 for eb_solver in solvers]
        temp_sim = [calc_apparent_temperature(eb_solver=eb_solver, date_obs=date_obs) for eb_solver in solvers]

        all_sim.append(temp_sim)
        all_obs.append(temp_obs)

        ax = plots.compare_temperature(obs=temp_obs, sim=temp_sim, ax=ax, return_ax=True,
                                       plot_colorbar=row_index == crop_df.index[-1])
        x_ls = range(24)
        ax2[0, 0].plot(x_ls, crop_weather['incident_diffuse_par_irradiance'], label=r'$\mathregular{{PAR}_{diff}}$')
        ax2[0, 0].plot(x_ls, crop_weather['incident_direct_par_irradiance'], label=r'$\mathregular{{PAR}_{dir}}$')
        ax2[0, 0].set_ylim((0, 500))
        ax2[0, 0].legend()

        ax2[1, 0].plot(x_ls, crop_weather['wind_speed'] / 3600., label='u')
        ax2[1, 0].set_ylim(0, 6)
        ax2[1, 0].legend()

        ax2[2, 0].plot(x_ls, crop_weather['air_temperature'], label=r'$\mathregular{T_{air}}$', color='black', linestyle='--')
        ax2[2, 0].plot(x_ls, temp_obs, label=r'$\mathregular{T_{can,\/obs}}$', color='orange')
        ax2[2, 0].plot(x_ls, temp_sim, label=r'$\mathregular{T_{can,\/sim}}$', color='blue')
        ax2[2, 0].set_ylim(-5, 60)
        ax2[2, 0].legend(fontsize='x-small')

        ax2[0, 1].text(0.1, 0.8, f'height={row["pl.height"] / 100.}', transform=ax2[0, 1].transAxes)
        ax2[0, 1].text(0.1, 0.6, f'LAI={row["LAI"]}', transform=ax2[0, 1].transAxes)

        ax2[1, 1].plot(x_ls, crop_weather['vapor_pressure_deficit'], label='VPD')
        ax2[1, 1].set_ylim(0, 6)
        ax2[1, 1].legend()

        ax2[2, 1].plot(x_ls, [eb_solver.crop.inputs.soil_water_potential for eb_solver in solvers], label=r'$\mathregular{\Psi_{soil}}$')
        ax2[2, 1].legend()

        ax2[0, 2] = plots.compare_temperature(obs=temp_obs, sim=temp_sim, ax=ax2[0, 2], return_ax=True)
        ax2[0, 2].text(0.95, 0.8, 'sim vs obs', ha='right', transform=ax2[0, 2].transAxes)

        fig2.savefig(path_figs / f'{row["Trt"]}{row["Plot"]}{date_obs.date()}.png')
        pyplot.close(fig2)

    all_sim = [x for sublist in all_sim for x in sublist]
    all_obs = [x for sublist in all_obs for x in sublist]
    ax.text(0.1, 0.9, f"R2 = {stats.calc_r2(all_sim, all_obs):.3f}", transform=ax.transAxes)
    ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(all_sim, all_obs):.3f}", transform=ax.transAxes)

    fig.savefig(path_figs / 'sim_vs_obs.png')
