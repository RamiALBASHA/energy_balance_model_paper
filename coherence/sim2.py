from pathlib import Path

from crop_energy_balance.formalisms import irradiance
from crop_energy_balance.formalisms.leaf import calc_stomatal_conductance
from crop_energy_balance.params import Constants
from crop_energy_balance.utils import discretize_linearly

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
        energy_balance_solver, _ = sim.solve_energy_balance(
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
        energy_balance_solver, _ = sim.solve_energy_balance(
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

    layers = {}
    irradiance = {}
    irradiance_object = {}
    temperature = {}
    solver_group = {}
    execution_time = {}

    for canopy_type, leaves_type in canopy_representations:
        print('-' * 50)
        print(f"{canopy_type} - {leaves_type}")
        canopy_layers = {0: sum(leaf_layers.values())} if canopy_type == 'bigleaf' else leaf_layers
        layers.update({f'{canopy_type} {leaves_type}': canopy_layers})
        hourly_absorbed_irradiance = []
        hourly_irradiance_obj = []
        hourly_temperature = []
        hourly_solver = []
        hourly_exe_time = []

        for date, w_data in weather_data.iterrows():
            print(date)
            absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
                leaf_layers=leaf_layers,
                is_bigleaf=(canopy_type == 'bigleaf'),
                is_lumped=(leaves_type == 'lumped'),
                incident_direct_par_irradiance=w_data['incident_direct_irradiance'],
                incident_diffuse_par_irradiance=w_data['incident_diffuse_irradiance'],
                solar_inclination_angle=w_data['solar_declination'])
            energy_balance_solver, exe_time = solve_energy_balance(
                vegetative_layers=canopy_layers,
                leaf_class_type=leaves_type,
                absorbed_par_irradiance=absorbed_irradiance,
                actual_weather_data=w_data,
                correct_stability=correct_for_stability)

            hourly_absorbed_irradiance.append(absorbed_irradiance)
            hourly_irradiance_obj.append(irradiance_obj)
            hourly_solver.append(energy_balance_solver)
            hourly_exe_time.append(exe_time)
            hourly_temperature.append(get_variable(
                var_to_get='temperature',
                one_step_solver=energy_balance_solver,
                leaf_class_type=leaves_type))

        irradiance[f'{canopy_type}_{leaves_type}'] = hourly_absorbed_irradiance
        irradiance_object[f'{canopy_type}_{leaves_type}'] = hourly_irradiance_obj
        temperature[f'{canopy_type}_{leaves_type}'] = hourly_temperature
        solver_group[f'{canopy_type}_{leaves_type}'] = hourly_solver
        execution_time[f'{canopy_type}_{leaves_type}'] = hourly_exe_time

    plots.plot_leaf_profile(vegetative_layers=layers, figure_path=figures_path)

    plots.plot_irradiance_dynamic_comparison(
        incident_irradiance=(weather_data.loc[:, ['incident_direct_irradiance', 'incident_diffuse_irradiance']]).sum(
            axis=1), all_cases_absorbed_irradiance=irradiance, figure_path=figures_path)

    plots.plot_temperature_dynamic_comparison(temperature_air=weather_data.loc[:, 'air_temperature'],
                                              all_cases_temperature=temperature, figure_path=figures_path)

    plots.plot_dynamic_comparison(
        solvers=solver_group, variable_to_plot='source_temperature', ylim=(5, 40), figure_path=figures_path)
    plots.plot_dynamic_comparison(
        solvers=solver_group, variable_to_plot='total_penman_monteith_evaporative_energy', ylim=(0, 900),
        figure_path=figures_path)

    for hour in range(24):
        plots.plot_temperature_one_hour_comparison2(hour=hour, hourly_weather=weather_data,
                                                    all_cases_absorbed_irradiance=(irradiance, irradiance_object),
                                                    all_cases_temperature=temperature, figure_path=figures_path)

    plots.plot_energy_balance(solvers=solver_group, figure_path=figures_path, plot_iteration_nb=True)

    plots.plot_stability_terms(solvers=solver_group, figs_path=figures_path)

    if correct_for_stability:
        plots.plot_universal_functions(solvers=solver_group, measurement_height=2, figure_path=figures_path)

    plots.plot_execution_time(execution_time_data=execution_time, figure_path=figures_path)

    plots.plot_properties_profile(
        solver_data=solver_group['layered_sunlit-shaded'],
        hours=[14],
        add_muliplicaiton_ax=True,
        component_props=['available_energy-penman_monteith_evaporative_energy', 'boundary_resistance'],
        multiply_by=[1, 1 / (Constants().air_density * Constants().air_specific_heat_capacity)],
        xlabels=[None, r'$\mathregular{\frac{r_a}{\rho\/C_{p}}\/[-]}$',
                 r'$\mathregular{\frac{r_a}{\rho\/C_{p}}\/({A\/-\/\lambda E})\/[^\circ C]}$'],
        figure_path=figures_path)
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


def demonstrate_surface_conductance_conceptual_difference():
    figs_path = Path(__file__).parents[1] / 'figs/coherence'
    figs_path.mkdir(exist_ok=True, parents=True)
    leaves_categories = ('sunlit', 'shaded', 'lumped')
    surface_conductance = {}
    for leaves_category in leaves_categories:
        leaf_surface_conductance_per_layer = []
        for cumulative_leaf_area_index in discretize_linearly(0, 4, 10):
            absorbed_irradiance = irradiance.calc_absorbed_irradiance(
                leaves_category=leaves_category,
                incident_direct_irradiance=225.16396909386614,
                incident_diffuse_irradiance=96.16936423946721,
                cumulative_leaf_area_index=cumulative_leaf_area_index,
                leaf_scattering_coefficient=0.15,
                canopy_reflectance_to_direct_irradiance=0.039585456879637215,
                canopy_reflectance_to_diffuse_irradiance=0.057,
                direct_extinction_coefficient=0.9121479255999952,
                direct_black_extinction_coefficient=0.9893633354937225,
                diffuse_extinction_coefficient=0.6204302130699697)

            lumped_leaf_surface_conductance = calc_stomatal_conductance(
                residual_stomatal_conductance=4.0,
                maximum_stomatal_conductance=39.6,
                absorbed_irradiance=absorbed_irradiance,
                shape_parameter=105,
                stomatal_sensibility_to_water_status=0.6274303009403469)

            if leaves_category == 'lumped':
                leaf_fraction = 1
            else:
                leaf_fraction = irradiance.calc_leaf_fraction(
                    leaves_category=leaves_category,
                    cumulative_leaf_area_index=cumulative_leaf_area_index,
                    direct_black_extinction_coefficient=0.9893633354937225)

            leaf_surface_conductance_per_layer.append(
                (lumped_leaf_surface_conductance * leaf_fraction, cumulative_leaf_area_index))
        surface_conductance.update({leaves_category: leaf_surface_conductance_per_layer})

    plots.plot_surface_conductance_profile(
        surface_conductance=surface_conductance,
        figure_path=figs_path)
    pass


def examine_soil_humidity_effect():
    figs_path = Path(__file__).parents[1] / 'figs/coherence'
    figs_path.mkdir(exist_ok=True, parents=True)

    leaf_class_type = 'sunlit-shaded'
    w_data = get_sq2_weather_data(filename='weather_maricopa_sunny.csv').loc[13]

    canopy_layers = {3: 1.0, 2: 1.0, 1: 1.0, 0: 1.0}
    saturation_ratios = [v / 100 for v in range(0, 110, 10)]
    absorbed_irradiance, _ = sim.calc_absorbed_irradiance(
        leaf_layers=canopy_layers,
        is_bigleaf=False,
        is_lumped=leaf_class_type == 'lumped',
        incident_direct_par_irradiance=w_data['incident_direct_irradiance'],
        incident_diffuse_par_irradiance=w_data['incident_diffuse_irradiance'],
        solar_inclination_angle=w_data['solar_declination'])

    temperature_ls = []
    latent_heat_ls = []
    for saturation_ratio in saturation_ratios:
        print(saturation_ratio)
        energy_balance_solver, _ = sim.solve_energy_balance(
            vegetative_layers=canopy_layers,
            leaf_class_type=leaf_class_type,
            absorbed_par_irradiance=absorbed_irradiance,
            actual_weather_data=w_data,
            correct_stability=False,
            inputs_update={"soil_saturation_ratio": saturation_ratio})
        temperature_ls.append(
            (saturation_ratio, energy_balance_solver.crop.state_variables.source_temperature - 273.15))
        latent_heat_ls.append(
            (saturation_ratio, energy_balance_solver.crop.state_variables.total_penman_monteith_evaporative_energy))

    plots.examine_soil_saturation_effect(temperature=temperature_ls, latent_heat=latent_heat_ls,
                                         figure_path=figs_path)


def examine_shift_effect():
    figs_path = Path(__file__).parents[1] / 'figs/coherence'
    figs_path.mkdir(exist_ok=True, parents=True)

    leaf_class_type = 'lumped'
    w_data = get_sq2_weather_data(filename='weather_maricopa_sunny.csv').loc[13]

    canopy_layers = {3: 1.0, 2: 1.0, 1: 1.0, 0: 1.0}
    saturation_ratios = [v / 100 for v in range(0, 125, 25)]
    absorbed_irradiance, _ = sim.calc_absorbed_irradiance(
        leaf_layers=canopy_layers,
        is_bigleaf=False,
        is_lumped=leaf_class_type == 'lumped',
        incident_direct_par_irradiance=w_data['incident_direct_irradiance'],
        incident_diffuse_par_irradiance=w_data['incident_diffuse_irradiance'],
        solar_inclination_angle=w_data['solar_declination'])

    temperature_ls = []
    for saturation_ratio in saturation_ratios:
        print(saturation_ratio)
        energy_balance_solver, _ = sim.solve_energy_balance(
            vegetative_layers=canopy_layers,
            leaf_class_type=leaf_class_type,
            absorbed_par_irradiance=absorbed_irradiance,
            actual_weather_data=w_data,
            correct_stability=False,
            inputs_update={"soil_saturation_ratio": saturation_ratio})
        temperature_ls.append(
            (saturation_ratio,
             sim.get_variable(
                 var_to_get='temperature',
                 one_step_solver=energy_balance_solver,
                 leaf_class_type=leaf_class_type)))

    plots.examine_shift_effect(lumped_temperature_ls=temperature_ls, figure_path=figs_path)


if __name__ == '__main__':
    run_four_canopy_sims()
    examine_diffuse_ratio_effect()
    examine_lai_effect()
    examine_soil_humidity_effect()
    examine_shift_effect()
    demonstrate_surface_conductance_conceptual_difference()
