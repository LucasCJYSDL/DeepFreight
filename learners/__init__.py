from .qmix_learner import Qmix_learner
from .wqmix_learner import WQmix_learner
from .coma_learner import COMALearner
from .msac_learner import SACQLearner
from .maven_learner import MAVENLearner

REGISTRY = {}
REGISTRY["qmix_learner"] = Qmix_learner
REGISTRY["wqmix_learner"] = WQmix_learner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["msac_learner"] = SACQLearner
REGISTRY["maven_learner"] = MAVENLearner

