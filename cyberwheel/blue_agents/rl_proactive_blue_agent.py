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
    The purpose of this blue agent is to prevent having to create new blue agents everytime a new 
    blue action is introduced. The idea is to have a config file specify what blue actions this instance
    has and import them dynamically.

    Actions need to be very standardized. Each one will need to have the following associated with it:
    - An action name: The name of the action performed. If you have two deploy actions, then the names would
    be something like: decoy0 and decoy1. Used by the reward calculator to determine reward.
    - A unique ID: Recurring rewards need an ID to identify them from other recurring actions. A UUID should
    be sufficient for this. If an action has no recurring cost (i.e. 0) then the ID can be "".

    This agent should also keep track of blue action config files. The config for decoys is an example.
    """
    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args)
    
    def get_observation_space(self, red_agent_result, headstart: bool) -> Iterable:
        alerts = self.observation.detector.obs([red_agent_result.action_results.detector_alert])
        return self.observation.create_obs_vector(alerts, headstart, self.network.get_num_decoys())