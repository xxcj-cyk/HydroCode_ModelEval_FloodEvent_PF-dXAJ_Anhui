from .API.API import api
from .API.API_CemaNeige import api_cemaneige
from .GR4J.GR4J import gr4j
from .GR4J.GR4J_CemaNeige import gr4j_cemaneige
from .GR4J.GR4J_mz import gr4j_mz
from .GR4J.GR4J_mz_CemaNeige import gr4j_mz_cemaneige
from .GR5J.GR5J import gr5j
from .GR5J.GR5J_CemaNeige import gr5j_cemaneige
from .GR5J.GR5J_mz import gr5j_mz
from .GR5J.GR5J_mz_CemaNeige import gr5j_mz_cemaneige
from .GR6J.GR6J import gr6j
from .GR6J.GR6J_CemaNeige import gr6j_cemaneige
from .GR6J.GR6J_mz import gr6j_mz
from .GR6J.GR6J_mz_CemaNeige import gr6j_mz_cemaneige
from .HBV.HBV import hbv
from .HBV.HBV_CemaNeige import hbv_cemaneige
from .HBV.HBV_mz import hbv_mz
from .HBV.HBV_mz_CemaNeige import hbv_mz_cemaneige
from .HYMOD.HYMOD import hymod
from .HYMOD.HYMOD_CemaNeige import hymod_cemaneige
from .TANK.TANK import tank
from .TANK.TANK_CemaNeige import tank_cemaneige
from .XAJ.XAJ import xaj
from .XAJ.XAJ_mz import xaj_mz
from .XAJ.XAJ_mz_CemaNeige import xaj_mz_cemaneige

MODEL_DICT = {
    'API': api,
    'API-CemaNeige': api_cemaneige,
    'GR4J': gr4j,
    'GR4J-CemaNeige': gr4j_cemaneige,
    'GR4J-mz': gr4j_mz,
    'GR4J-mz-CemaNeige': gr4j_mz_cemaneige,
    'GR5J': gr5j,
    'GR5J-CemaNeige': gr5j_cemaneige,
    'GR5J-mz': gr5j_mz,
    'GR5J-mz-CemaNeige': gr5j_mz_cemaneige,
    'GR6J': gr6j,
    'GR6J-CemaNeige': gr6j_cemaneige,
    'GR6J-mz': gr6j_mz,
    'GR6J-mz-CemaNeige': gr6j_mz_cemaneige,
    'HBV': hbv,
    'HBV-CemaNeige': hbv_cemaneige,
    'HBV-mz': hbv_mz,
    'HBV-mz-CemaNeige': hbv_mz_cemaneige,
    'HYMOD': hymod,
    'HYMOD-CemaNeige': hymod_cemaneige,
    'TANK': tank,
    'TANK-CemaNeige': tank_cemaneige,
    'XAJ': xaj,
    'XAJ-mz': xaj_mz,
    'XAJ-mz_Anhui': xaj_mz,
    'XAJ-mz-CemaNeige': xaj_mz_cemaneige,
}
