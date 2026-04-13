from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_continuous_learner import ActorCriticContinuousLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .ppo_continuous_learner import PPOContinuousLearner
from .magrpo_continuous_learner import MAGRPOContinuousLearner
from .maddpg_continuous_learner import MADDPGContinuousLearner


REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["actor_critic_continuous_learner"] = ActorCriticContinuousLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["ppo_continuous_learner"] = PPOContinuousLearner
REGISTRY["magrpo_continuous_learner"] = MAGRPOContinuousLearner
REGISTRY["maddpg_continuous_learner"] = MADDPGContinuousLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
