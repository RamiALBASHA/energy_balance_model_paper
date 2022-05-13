import datetime
from datetime import timedelta

from crop_energy_balance.solver import Solver
from pandas import read_excel

from sim_vs_obs.maricopa_hsc import plots
from sim_vs_obs.maricopa_hsc.base_functions import get_weather_data, set_energy_balance_inputs
from sim_vs_obs.maricopa_hsc.config import PathInfos, WeatherInfo, ParamsInfo

if __name__ == '__main__':
    is_stability_corrected = True
    crop_representation = 'bigleaf_sunlit-shaded'

    is_bigleaf = 'bigleaf' in crop_representation
    is_lumped = 'lumped' in crop_representation

    path_source_raw = PathInfos.source_raw.value
    path_source_fmt = PathInfos.source_fmt.value

    outputs_dir = 'corrected' if is_stability_corrected else 'neutral'
    path_outputs = PathInfos.source_fmt.value.parent / 'outputs' / outputs_dir / crop_representation
    path_outputs.mkdir(parents=True, exist_ok=True)

    if is_bigleaf:
        lai_distribution_ratios = [1]
    else:
        lai_distribution_ratios = ParamsInfo().lai_distribution_ratios

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

    all_solvers = {d: {} for d in set(list(crop_df['date']))}
    for row_index, row in crop_df.iterrows():
        print(row_index)
        leaf_layers = {0: row['LAI']} if is_bigleaf else {i: row['LAI'] * r
                                                          for i, r in enumerate(lai_distribution_ratios)}
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
                is_lumped=is_lumped,
                datetime_obs=datetime_obs,
                crop_data=row,
                soil_data=soil_df,
                weather_data=w_data)

            solver = Solver(leaves_category='lumped' if is_lumped else 'sunlit_shaded',
                            inputs_dict=eb_inputs,
                            params_dict=eb_params)
            solver.run(is_stability_considered=is_stability_corrected)
            solvers.append(solver)

        all_solvers[date_obs].update({
            f'plot_{row["Plot"]}': {
                'solvers': solvers,
                'temp_obs': crop_weather['canopy_temperature'].tolist()}})

    plots.plot_dynamic(all_solvers=all_solvers, path_figs=path_outputs)

    results = plots.extract_results(all_solvers=all_solvers)
    plots.plot_summary(results=results, path_figs=path_outputs)
    plots.plot_errors(all_solvers=all_solvers, path_figs=path_outputs)
    plots.export_results(results=results, path_csv=path_outputs)
