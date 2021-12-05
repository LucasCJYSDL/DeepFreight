REGISTRY = {}

from .rnn_agent import RNNAgent
from .central_rnn_agent import CentralRNNAgent
from .noise_rnn_agent import NoiseRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY['noise_rnn'] = NoiseRNNAgent
