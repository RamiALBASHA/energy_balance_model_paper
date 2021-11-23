from json import load

import matplotlib.pyplot as plt
from SALib.analyze import fast
from SALib.sample import fast_sampler
from crop_energy_balance.solver import Solver
from numpy import array, arange

from coherence.plots import UNITS_MAP

MAP_PARAMS = {
    'vpd_coeff': r'$\mathregular{D_o}$',
    'soil_aerodynamic_resistance_shape_parameter': r'$\mathregular{\alpha_w}$',
    'soil_roughness_length_for_momentum': r'$\mathregular{z_{0,\/u}}}$',
    'leaf_characteristic_length': r'$\mathregular{w}$',
    'leaf_boundary_layer_shape_parameter': r'$\mathregular{\alpha}$',
    'wind_speed_extinction_coef': r'$\mathregular{k_u}$',
    'maximum_stomatal_conductance': r'$\mathregular{g_{s,\/max}}$',
    'residual_stomatal_conductance': r'$\mathregular{g_{s,\/res}}$',
    'diffuse_extinction_coef': r'$\mathregular{k_{diffuse}}$',
    'leaf_scattering_coefficient': r'$\mathregular{\sigma_s}$',
    'absorbed_par_50': r'$\mathregular{R_{l,\/PAR,\/50}}$',
    'soil_resistance_to_vapor_shape_parameter_1': r'$\mathregular{a_s}$',
    'soil_resistance_to_vapor_shape_parameter_2': r'$\mathregular{b_s}$'
}


def eb_wrapper(inputs: dict, params: dict) -> Solver:
    solver = Solver(leaves_category='lumped',
                    inputs_dict=inputs,
                    params_dict=params)
    solver.run()
    return solver


def sample(config_path: str) -> (dict, array):
    with open(config_path, mode='r') as f:
        param_fields = load(f)
    names, bounds = zip(*list(param_fields.items()))
    problem = {
        'num_vars': len(names),
        'names': list(names),
        'bounds': list(bounds)
    }
    return problem, fast_sampler.sample(problem=problem, N=2 ** 7)


def evaluate(inputs: dict, params: dict, names: list, scenarios: array, output_variables: list):
    res = {k: [] for k in output_variables}
    for i, scenario in enumerate(scenarios):
        params.update({k: v for k, v in zip(names, scenario)})
        solver = eb_wrapper(inputs=inputs, params=params)
        for k in output_variables:
            res[k].append(getattr(solver.crop.state_variables, k))
        print(i)
    return {k: array(v) for k, v in res.items()}


def analyze(problem: dict, outputs: array) -> dict:
    return {k: fast.analyze(problem, v, M=4, num_resamples=100, conf_level=0.95, print_to_console=False, seed=None)
            for k, v in outputs.items()}


def plot(sa_dict: {str, dict}, shift_bars: bool = False, suptitle: str = None):
    fig, axs = plt.subplots(ncols=len(sa_dict.keys()))
    bar_height = 0.4

    for ax, (k, sa) in zip(axs, sa_dict.items()):
        plot_single(ax, bar_height, k, sa, shift_bars=shift_bars)

    axs[0].legend()
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(r'../figs/sensitivity_analysis.png')
    plt.close()
    pass


def plot_single(ax, bar_height, k, sa, shift_bars=True):
    sa_df = sa.to_df()
    sa_df.sort_values(by='S1', inplace=True)
    y_pos = arange(len(sa['names']))
    ax.clear()
    if shift_bars:
        for effect, shift in (('S1', bar_height / 2), ('ST', -bar_height / 2)):
            ax.barh(y_pos + shift, sa_df[effect].values, height=bar_height, label=effect,
                    xerr=sa_df[f'{effect}_conf'].values,
                    error_kw=dict(ecolor='gray', lw=0.5, capsize=2, capthick=0.5))
    else:
        ax.barh(y_pos, sa_df['S1'].values, height=bar_height, label='S1')
        ax.barh(y_pos, sa_df['ST'].values, height=bar_height, label='ST', left=sa_df['S1'].values)

    ax.set(yticks=y_pos + 0.2, yticklabels=[MAP_PARAMS[s] for s in sa_df.index], title=UNITS_MAP[k][0])
    pass


if __name__ == '__main__':
    input_scenario = 'inputs.json'
    with open(f'sources/{input_scenario}', mode='r') as f:
        base_inputs = load(f)
    with open(r'sources/params.json', mode='r') as f:
        base_params = load(f)
    problem, param_values = sample(config_path='sources/param_fields.json')
    outputs = evaluate(inputs=base_inputs, params=base_params, names=problem['names'], scenarios=param_values,
                       output_variables=['source_temperature', 'total_penman_monteith_evaporative_energy'])
    sa_result = analyze(problem=problem, outputs=outputs)
    plot(sa_dict=sa_result, shift_bars=False, suptitle=input_scenario)
