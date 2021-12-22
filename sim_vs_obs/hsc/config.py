from enum import Enum
from pathlib import Path

from crop_energy_balance.formalisms.leaf import calc_stomatal_sensibility_misson
from pandas import read_excel

from utils.van_genuchten_params import VanGenuchtenParams


class PathInfos(Enum):
    source_raw = Path(__file__).parents[2] / 'sources/hsc/data_raw'
    source_fmt = Path(__file__).parents[2] / 'sources/hsc/data_fmt'


class WeatherInfo(Enum):
    measurement_height = 3.
    reference_height = 2.
    atmospheric_pressure = 101.3
    latitude = 33.069
    longitude = -114.53
    elevation = 361


class SoilInfo(Enum):
    Vol_con_10cm = 0.15
    Vol_con_20cm = 0.10
    Vol_con_30cm = 0.15
    Vol_con_50cm = 0.25
    Vol_con_80cm = 0.35
    soil_class = 'Sandy_Clay_Loam'
    hydraulic_props = VanGenuchtenParams.Sandy_Clay_Loam.value


class ParamsInfo:
    def __init__(self):
        self.psi_half_aperture = -0.9703454766964716

    def fit_psi_half_aperture(self, plot_result: bool = False):
        """Calculates a rough estimation of Misson's 'half_stomatal_aperture' parameters

        Args:
            plot_result: plot data and fitted curve if True

        Returns:

        """
        from scipy.optimize import curve_fit

        raw_df = read_excel(PathInfos.source_raw.value / '3. Crop response data/Gas_Exchange_and_Water_Relations.ods',
                            sheet_name='Data', engine='odf')
        raw_df.dropna(inplace=True)
        sub_df = raw_df[(raw_df['TRT'] != 'H') & (raw_df['gs'] >= 0.3)]  # rough filtering of light-saturated gs
        water_potential_values = sub_df['TLWP']
        stomatal_aperture_values = sub_df['gs']
        popt, pcov = curve_fit(calc_stomatal_sensibility_misson, xdata=water_potential_values,
                               ydata=stomatal_aperture_values,
                               bounds=([-3., 2.], [-0.1, 2.01]), sigma=sub_df['gs.stdev'])

        if plot_result:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
            sim = [calc_stomatal_sensibility_misson(v, *popt) for v in sorted(water_potential_values)]
            ax.scatter(water_potential_values, stomatal_aperture_values, label='obs (C & R)')
            ax.plot(sorted(water_potential_values), sim, label='sim (C & R)')
            ax.set(xlabel='water potential [MPa]', ylabel='reduction factor [-]')
            ax.legend(loc='upper left')
            fig.savefig('stomatal_sensitivity_fit.png')

        self.psi_half_aperture = popt[0]
