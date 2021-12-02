from enum import Enum


class VanGenuchtenParams(Enum):
    """Default soil hydrodynamic parameters for the model of van Genuchten-Muallem.
    For each soil class the data are organized as follows:
        (theta_r, theta_s, alpha[cm-1], n, k_sat[cm d-1], m=1-1/n)

    References
        J. Simunek, M. Sejna, H. Saito, M. Sakai, and M. Th. van Genuchten, 2013.
            The HYDRUS-1D Software Package for Simulating the One-Dimensional Movement of Water, Heat, and Multiple
                Solutes in Variably-Saturated Media. Version 4.17
                Department of Environmental Sciences. University of California Riverside, Riverside, California.
        Carsel R., Parrish R., 1988.
            Developing joint probability distributions of soil water retention characteristics.
            Water Resources Research 24,755 – 769.
    """

    Sand = [0.045, 0.430, 0.145, 2.68, 712.8, 0.626865672]
    Loamy_Sand = [0.057, 0.410, 0.124, 2.28, 350.2, 0.561403509]
    Sandy_Loam = [0.065, 0.410, 0.075, 1.89, 106.1, 0.470899471]
    Loam = [0.078, 0.430, 0.036, 1.56, 24.96, 0.358974359]
    Silt = [0.034, 0.460, 0.016, 1.37, 6.00, 0.270072993]
    Silty_Loam = [0.067, 0.450, 0.020, 1.41, 10.80, 0.290780142]
    Sandy_Clay_Loam = [0.100, 0.390, 0.059, 1.48, 31.44, 0.324324324]
    Clay_Loam = [0.095, 0.410, 0.019, 1.31, 6.24, 0.236641221]
    Silty_Clay_Loam = [0.089, 0.430, 0.010, 1.23, 1.68, 0.18699187]
    Sandy_Clay = [0.100, 0.380, 0.027, 1.23, 2.88, 0.18699187]
    Silty_Clay = [0.070, 0.360, 0.005, 1.09, 0.48, 0.082568807]
    Clay = [0.068, 0.380, 0.008, 1.09, 4.80, 0.082568807]
