from pathlib import Path

from coherence import sim, plots
from sources.demo import get_sq2_weather_data


def compare_sunlit_shaded_temperatures():
    """Examines the effect of incident irradiance diffusion ratio on the temperature difference between sunlit and
    shaded leaves.
    """
    canopy_layers = {3: 1.0, 2: 1.0, 1: 1.0, 0: 1.0}
    leaf_class_type = 'sunlit-shaded'

    weather_data = get_sq2_weather_data(filename='weather_maricopa_sunny.csv')
    weather_data.drop(index=[i for i in weather_data.index if i != 13], inplace=True)

    incident_irradiance = weather_data.loc[13, ['incident_direct_irradiance', 'incident_diffuse_irradiance']].sum()

    weather_data = weather_data.loc[weather_data.index.repeat(11)]
    weather_data.reset_index(drop=True, inplace=True)
    weather_data.loc[:, 'diffuse_ratio'] = [i / 10 for i in range(11)]
    weather_data.loc[10, 'diffuse_ratio'] = 0.99
    weather_data.loc[:, 'incident_direct_irradiance'] = weather_data.apply(
        lambda x: incident_irradiance * (1 - x['diffuse_ratio']), axis=1)
    weather_data.loc[:, 'incident_diffuse_irradiance'] = weather_data.apply(
        lambda x: incident_irradiance * x['diffuse_ratio'], axis=1)

    hourly_temperature = []
    for i, w_data in weather_data.iterrows():
        print(i)
        absorbed_irradiance, _ = sim.calc_absorbed_irradiance(
            leaf_layers=canopy_layers,
            is_bigleaf=False,
            is_lumped=leaf_class_type == 'lumped',
            incident_direct_par_irradiance=w_data['incident_direct_irradiance'],
            incident_diffuse_par_irradiance=w_data['incident_diffuse_irradiance'],
            solar_inclination_angle=w_data['solar_declination'])
        energy_balance_solver = sim.solve_energy_balance(
            vegetative_layers=canopy_layers,
            leaf_class_type=leaf_class_type,
            absorbed_par_irradiance=absorbed_irradiance,
            actual_weather_data=w_data,
            correct_stability=False)
        hourly_temperature.append(
            (w_data['diffuse_ratio'],
             sim.get_variable(
                 var_to_get='temperature',
                 one_step_solver=energy_balance_solver,
                 leaf_class_type=leaf_class_type)))

    plots.compare_sunlit_shaded_temperatures(hourly_temperature=hourly_temperature,
                                             figure_path=Path(__file__).parents[1] / 'figs/coherence')

    pass


if __name__ == '__main__':
    compare_sunlit_shaded_temperatures()
