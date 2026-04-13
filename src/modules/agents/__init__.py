from .rnn_agent import RNNAgent
from .rnn_gaussian_agent import RNNGaussianAgent
from .transformer_gaussian_agent import TransformerGaussianAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_gaussian"] = RNNGaussianAgent
REGISTRY["transformer_gaussian"] = TransformerGaussianAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
