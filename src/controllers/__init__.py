REGISTRY = {}

from .basic_controller import BasicMAC
from .gaussian_controller import GaussianMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .maddpg_continuous_controller import MADDPGContinuousMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["gaussian_mac"] = GaussianMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["maddpg_continuous_mac"] = MADDPGContinuousMAC
