import builtins
import importlib
import yaml

from importlib.resources import files
from typing import Dict, List, Any, Iterable
from gymnasium import Space

from cyberwheel.blue_agents.blue_agent import BlueAgent, BlueAgentResult
from cyberwheel.blue_agents.rl_blue_agent import host_to_index_mapping, _ActionConfigInfo, RLBlueAgent
from cyberwheel.reward.reward_base import RewardMap
from cyberwheel.network.network_base import Network, Host
from cyberwheel.blue_agents.action_space.action_space import ActionSpace

class RLBlueAgentProactive(RLBlueAgent):
    """
    This proactive blue agent deploys a set amount of decoys before the simulated cyber attack occurs.

    The observation space includes two more additional attributes:
    1. The number of decoys currently deployed.
    2. If the agent is in the headstart phase or not.

    We pass these additional attributes to BlueObservationProactive, where it appends to the end of the observation space.
    """
    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args)
    
    def get_observation_space(self, red_agent_result, headstart: bool) -> Iterable:
        alerts = self.observation.detector.obs([red_agent_result.action_results.detector_alert])
        return self.observation.create_obs_vector(alerts, headstart, self.network.get_num_decoys())