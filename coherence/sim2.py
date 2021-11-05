from pathlib import Path

from coherence import sim, plots
from coherence.sim import calc_absorbed_irradiance, solve_energy_balance, get_variable
from coherence.sources.demo import get_sq2_weather_data, plot_weather


def examine_diffuse_ratio_effect():
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

    temperature_ls = []
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
        temperature_ls.append(
            (w_data['diffuse_ratio'],
             sim.get_variable(
                 var_to_get='temperature',
                 one_step_solver=energy_balance_solver,
                 leaf_class_type=leaf_class_type)))

    plots.compare_sunlit_shaded_temperatures(
        temperature_data=temperature_ls,
        figure_path=Path(__file__).parents[1] / 'figs/coherence/effect_diffusion_ratio.png',
        xlabel='diffuse ratio [-]')

    pass


def examine_lai_effect():
    leaf_class_type = 'sunlit-shaded'
    w_data = get_sq2_weather_data(filename='weather_maricopa_sunny.csv').loc[13]
    layers_number = 4

    temperature_ls = []
    for lai in [0.1, 0.5] + list(range(1, 8)):
        print(lai)
        canopy_layers = {k: lai / layers_number for k in reversed(range(layers_number))}
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
        temperature_ls.append(
            (lai,
             sim.get_variable(
                 var_to_get='temperature',
                 one_step_solver=energy_balance_solver,
                 leaf_class_type=leaf_class_type)))

    plots.compare_sunlit_shaded_temperatures(
        temperature_data=temperature_ls,
        figure_path=Path(__file__).parents[1] / 'figs/coherence/effect_lai.png',
        xlabel=' '.join(plots.UNITS_MAP['LAI']))


def sim_general(canopy_representations: tuple, leaf_layers: dict, weather_file_name: str, correct_for_stability: bool,
                figures_path: Path):
    figures_path = figures_path / weather_file_name.split('.')[0]
    figures_path.mkdir(parents=True, exist_ok=True)
    weather_data = get_sq2_weather_data(weather_file_name)

    irradiance = {}
    irradiance_object = {}
    temperature = {}
    layers = {}
    solver_group = {}

    for canopy_type, leaves_type in canopy_representations:
        print('-' * 50)
        print(f"{canopy_type} - {leaves_type}")
        canopy_layers = {0: sum(leaf_layers.values())} if canopy_type == 'bigleaf' else leaf_layers
        layers.update({f'{canopy_type} {leaves_type}': canopy_layers})
        hourly_absorbed_irradiance = []
        hourly_irradiance_obj = []
        hourly_temperature = []
        hourly_solver = []

        for date, w_data in weather_data.iterrows():
            print(date)
            absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
                leaf_layers=leaf_layers,
                is_bigleaf=(canopy_type == 'bigleaf'),
                is_lumped=(leaves_type == 'lumped'),
                incident_direct_par_irradiance=w_data['incident_direct_irradiance'],
                incident_diffuse_par_irradiance=w_data['incident_diffuse_irradiance'],
                solar_inclination_angle=w_data['solar_declination'])
            energy_balance_solver = solve_energy_balance(
                vegetative_layers=canopy_layers,
                leaf_class_type=leaves_type,
                absorbed_par_irradiance=absorbed_irradiance,
                actual_weather_data=w_data,
                correct_stability=correct_for_stability)

            hourly_absorbed_irradiance.append(absorbed_irradiance)
            hourly_irradiance_obj.append(irradiance_obj)
            hourly_solver.append(energy_balance_solver)
            hourly_temperature.append(get_variable(
                var_to_get='temperature',
                one_step_solver=energy_balance_solver,
                leaf_class_type=leaves_type))

        irradiance[f'{canopy_type}_{leaves_type}'] = hourly_absorbed_irradiance
        irradiance_object[f'{canopy_type}_{leaves_type}'] = hourly_irradiance_obj
        temperature[f'{canopy_type}_{leaves_type}'] = hourly_temperature
        solver_group[f'{canopy_type}_{leaves_type}'] = hourly_solver

    plots.plot_leaf_profile(vegetative_layers=layers, figure_path=figures_path)

    plots.plot_irradiance_dynamic_comparison(
        incident_irradiance=(weather_data.loc[:, ['incident_direct_irradiance', 'incident_diffuse_irradiance']]).sum(
            axis=1), all_cases_absorbed_irradiance=irradiance, figure_path=figures_path)

    plots.plot_temperature_dynamic_comparison(temperature_air=weather_data.loc[:, 'air_temperature'],
                                              all_cases_temperature=temperature, figure_path=figures_path)

    for hour in range(24):
        plots.plot_temperature_one_hour_comparison2(hour=hour, hourly_weather=weather_data,
                                                    all_cases_absorbed_irradiance=(irradiance, irradiance_object),
                                                    all_cases_temperature=temperature, figure_path=figures_path)

    plots.plot_energy_balance(solvers=solver_group, figure_path=figures_path, plot_iteration_nb=True)

    plots.plot_stability_terms(solvers=solver_group, figs_path=figures_path)

    if correct_for_stability:
        plots.plot_universal_functions(solvers=solver_group, measurement_height=2, figure_path=figures_path)

    pass


def run_four_canopy_sims():
    figs_path = Path(__file__).parents[1] / 'figs/coherence'
    figs_path.mkdir(exist_ok=True, parents=True)
    weather_files = {'sunny': 'weather_maricopa_sunny.csv', 'cloudy': 'weather_maricopa_cloudy.csv'}
    plot_weather(
        weather_data={k: get_sq2_weather_data(v) for k, v in weather_files.items()},
        figure_path=figs_path / 'weather_maricopa.png')

    for weather_file in weather_files.values():
        sim_general(
            canopy_representations=(('bigleaf', 'lumped'),
                                    ('bigleaf', 'sunlit-shaded'),
                                    ('layered', 'lumped'),
                                    ('layered', 'sunlit-shaded')),
            leaf_layers={3: 1.0, 2: 1.0, 1: 1.0, 0: 1.0},
            weather_file_name=weather_file,
            correct_for_stability=False,
            figures_path=figs_path)
    pass


if __name__ == '__main__':
    run_four_canopy_sims()
    examine_diffuse_ratio_effect()
    examine_lai_effect()
