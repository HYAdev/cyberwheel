from cyberwheel.network.network_base import Host, Network
from cyberwheel.reward.reward_base import Reward, RewardMap, RecurringAction
from cyberwheel.reward.rl_reward import RLReward
from cyberwheel.utils.hybrid_set_list import HybridSetList

class RLRewardProactive(RLReward):
    """
    The proactive RL blue agent is rewarded for every timestep where the attacker targets one of its decoys.
    """

    def __init__(
        self,
        red_rewards: RewardMap,
        blue_rewards: RewardMap,
        args,
        network: Network
    ) -> None:
        super().__init__(red_rewards, blue_rewards, args.valid_targets, network)
        self.args = args

    def _DELAY(self, decoy):
        return 40.0 if decoy else 0
    
    def _GENERAL(self, decoy=False):
        return 0
    
    def calculate_reward(
        self,
        red_action: str,
        blue_action: str,
        red_success: str,
        blue_success: bool,
        target_host: Host,
        blue_id: str = -1,
        blue_recurring: int = 0,
        headstart : bool = False,
        num_impacted_decoys : int = 0,
    ) -> int | float:

        objective = self.args.objective
        post_play = self.args.post_play
        exceeded_decoy_limit = self.network.get_num_decoys() >= self.args.decoy_limit
        
        if self.valid_targets == "servers":
            valid_targets = self.network.server_hosts
        elif self.valid_targets == "users":
            valid_targets = self.network.user_hosts
        elif self.valid_targets == "all":
            valid_targets = valid_targets = HybridSetList(self.network.hosts.keys())
        elif type(self.valid_targets) is list:
            valid_targets = HybridSetList(self.valid_targets)
        elif type(self.valid_targets) is str:
            valid_targets = HybridSetList(self.valid_targets)
        else:
            valid_targets = HybridSetList(self.network.hosts.keys())

        target_host_name = target_host.name
        decoy = target_host.decoy

        r = 0
        r_recurring = 0

        multiplier = 1


        if blue_success and blue_action == "deploy_decoy":
            if exceeded_decoy_limit: # deployed decoy after limit
                multiplier = 3
            elif headstart:
                multiplier = -3 # deployed decoy in proper range
            else: # after headstart
                if post_play: # after headstart actions are allowed
                    multiplier = 1 # deployed decoy in proper range, after headstart
                else:
                    multiplier = 60 # deployed decoy after headstart when not allowed
        else: # if not deploying decoy, same reward
            multiplier = 1

        b = self.blue_rewards[blue_action][0] * multiplier

        action_dict = {
            "delay": self._DELAY,
            "general": self._GENERAL
        }

        if objective in action_dict:
            b += action_dict[objective](decoy) # currently only working with the DELAY reward
        
        if r_recurring != 0:
            self.add_recurring_red_action('0', red_action, decoy)

        if blue_recurring == -1:
            self.remove_recurring_blue_action(blue_id)
        elif blue_recurring == 1:
            self.add_recurring_blue_action(blue_id, blue_action)

        return r + b + self.sum_recurring()