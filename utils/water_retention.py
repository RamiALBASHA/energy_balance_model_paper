from utils.van_genuchten_params import VanGenuchtenParams


def calc_soil_water_potential(theta: float, soil_class: str) -> float:
    """Computes soil water potential following van Genuchten (1980)

    Args:
        theta: [-] volumetric soil water content
        soil_class (str): one of the soil classes proposed by Carsel and Parrish (1988), see :class:`VanGenuchtenParams`

    Returns:
        (float): [cm H2O] soil water potential

    Reference:
        van Genuchten M., 1980.
            A closed-form equation for predicting the hydraulic conductivity of unsaturated soils.
            Soil Science Society of America Journal 44, 892897.
    """

    theta_r, theta_s, alpha, n, k_sat, m = getattr(VanGenuchtenParams, soil_class).value
    theta = max(theta, theta_r * (1 + 1.e-6))
    if theta == theta_s:
        psi_soil = 0
    else:
        s_e = (theta - theta_r) / (theta_s - theta_r)
        psi_soil = - 1. / alpha * ((1. / s_e) ** (1. / m) - 1) ** (1. / n)

    return psi_soil


def calc_soil_water_content(psi: float, soil_class: str) -> float:
    """Computes soil water potential following van Genuchten (1980)

    Args:
        psi: [cm H2O] volumetric soil water content
        soil_class (str): one of the soil classes proposed by Carsel and Parrish (1988), see :class:`VanGenuchtenParams`

    Returns:
        (float): [cm H2O] soil water potential

    Reference:
        van Genuchten M., 1980.
            A closed-form equation for predicting the hydraulic conductivity of unsaturated soils.
            Soil Science Society of America Journal 44, 892897.
    """

    theta_r, theta_s, alpha, n, k_sat, m = getattr(VanGenuchtenParams, soil_class).value

    if psi == 0:
        theta = theta_s
    else:
        theta = theta_r + (theta_s - theta_r) / (1. + abs(alpha * psi) ** n) ** m

    return theta
