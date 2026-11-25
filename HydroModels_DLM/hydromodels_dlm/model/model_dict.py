from hydromodels_dlm.model.LSTM import SeqRegLSTM
from hydromodels_dlm.model.dGR4J import GR4JCore
from hydromodels_dlm.model.dGR4J_CemaNeige import GR4JCemaNeigeCore
from hydromodels_dlm.model.dXAJ import XAJCore
from hydromodels_dlm.model.dXAJ_mz import XAJMzAnhuiCore, XAJMzCore
from hydromodels_dlm.model.dXAJ_mz_CemaNeige import XAJMzCemaNeigeCore

MODEL_DICT = {
    'SeqRegLSTM': SeqRegLSTM,
    'dGR4J': GR4JCore,
    'dGR4J-CemaNeige': GR4JCemaNeigeCore,
    'dXAJ': XAJCore,
    'dXAJ-mz': XAJMzCore,
    'dXAJ-mz_Anhui': XAJMzAnhuiCore,
    'dXAJ-mz-CemaNeige': XAJMzCemaNeigeCore,
}
