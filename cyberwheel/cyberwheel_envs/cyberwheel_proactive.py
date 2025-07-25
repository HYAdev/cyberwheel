from cyberwheel.red_actions.actions.nothing import Nothing
from cyberwheel.red_agents.red_agent_base import RedAgentResult

from typing import Iterable, Any 

from cyberwheel.cyberwheel_envs.cyberwheel_rl import CyberwheelRL
from cyberwheel.network.network_base import Network
from cyberwheel.utils import YAMLConfig

class CyberwheelProactive(CyberwheelRL):
    metadata = {"render.modes": ["human"]}

    def __init__( self, args: YAMLConfig, network: Network = None, evaluation: bool = False):
        super().__init__(args, network=network, evaluation=evaluation)
        self.args = args

    def step(self, action: int) -> tuple[Iterable, int | float, bool, bool, dict[str, Any]]:
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Get obs from Red or Blue Observation
        5. Return obs and related metadata
        """

        in_headstart = self.current_step < self.args.decoy_limit

        blue_agent_result = self.blue_agent.act(action)

        if in_headstart:
            action_results = Nothing(self.red_agent.current_host, self.red_agent.current_host).sim_execute()
            red_agent_result = RedAgentResult(action_results.action, self.red_agent.current_host, self.red_agent.current_host, False, action_results=action_results)
        else:
            red_agent_result = self.red_agent.act(action)
        obs_vec = self.blue_agent.get_observation_space(red_agent_result, in_headstart)

        reward = self.reward_sign * self.reward_calculator.calculate_reward(
            red_agent_result.action.get_name(),
            blue_agent_result.name,
            red_agent_result.success,
            blue_agent_result.success,
            red_agent_result.target_host,
            blue_id=blue_agent_result.id,
            blue_recurring=blue_agent_result.recurring,
            headstart=in_headstart,
            num_impacted_decoys=self.network.get_num_compromised_decoys()
        )

        self.total += reward

        done = red_agent_result.action.get_name() == "impact"

        self.current_step += 1
        info = {}
        
        pingsweeped_decoy = False
        if (red_agent_result.action.get_name() == "pingsweep"):
            for host in red_agent_result.action_results.metadata.get("sweeped_hosts"):
                if host.decoy:
                    pingsweeped_decoy = True
        
        if self.evaluation:
            info = {
                "red_action": red_agent_result.action.get_name(),
                "red_action_src": red_agent_result.src_host.name,
                "red_action_dst": red_agent_result.target_host.name,
                "red_action_dst_is_decoy": red_agent_result.target_host.decoy,
                "red_action_success": red_agent_result.success,
                "blue_action": blue_agent_result.name,
                "blue_action_id": blue_agent_result.id,
                "blue_action_target": blue_agent_result.target,
                "killchain": self.red_agent.killchain,
                "network": self.network,
                "history": self.red_agent.history,
                "commands": red_agent_result.action_results.metadata
                    .get(red_agent_result.target_host.name, {})
                    .get("commands", []),
                "pingsweeped_decoy": pingsweeped_decoy,
                "decoy_attacked": red_agent_result.target_host.decoy,
                "impacted_decoys": self.network.get_num_compromised_decoys(),
                "timestep_till_impact": self.current_step if done else 0,
                "headstart": in_headstart
            }

        return obs_vec, reward, done, False, info