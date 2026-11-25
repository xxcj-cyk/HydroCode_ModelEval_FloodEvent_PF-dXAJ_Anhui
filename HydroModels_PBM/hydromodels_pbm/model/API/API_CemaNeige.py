import numpy as np

from hydromodels_pbm.model.API.API import simulate
from hydromodels_pbm.model.cemaneige import liquid_precip_and_pet, require_snow_drivers
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


def api_cemaneige(drivers, parameters, warmup_length=365):
    require_snow_drivers(drivers, 'API-CemaNeige')
    params = to_array_2d(parameters)
    p_and_e = liquid_precip_and_pet(
        drivers, params[:, 3], params[:, 4], warmup_length
    )
    qsim = simulate(p_and_e, params[:, 0], params[:, 1], params[:, 2])
    ets = np.zeros((drivers.shape[0], drivers.shape[1]), dtype=np.float64)
    return trim_warmup_output(qsim, ets, warmup_length)
