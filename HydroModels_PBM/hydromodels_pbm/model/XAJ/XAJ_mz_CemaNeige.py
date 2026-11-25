from hydromodels_pbm.model.XAJ.XAJ_mz import simulate_xaj_mz
from hydromodels_pbm.model.cemaneige import liquid_precip_and_pet, require_snow_drivers
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


def xaj_mz_cemaneige(drivers, parameters, warmup_length=365):
    require_snow_drivers(drivers, 'XAJ-mz-CemaNeige')
    params = to_array_2d(parameters)
    p_and_e = liquid_precip_and_pet(
        drivers, params[:, 15], params[:, 16], warmup_length
    )
    qsim, ets = simulate_xaj_mz(
        p_and_e,
        params[:, :15],
        warmup_length=warmup_length,
    )
    return trim_warmup_output(qsim, ets, warmup_length)
