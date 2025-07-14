from cyberwheel.network.network_base import Host, Network
from cyberwheel.reward.reward_base import Reward, RewardMap, RecurringAction
from cyberwheel.utils.hybrid_set_list import HybridSetList

class RLBaselineReward(Reward):
    def __init__(
        self,
        red_rewards: RewardMap,
        blue_rewards: RewardMap,
        args,
        network: Network
    ) -> None:
        super().__init__(red_rewards, blue_rewards)
        self.valid_targets = args.valid_targets
        self.network = network
        self.args = args

    def _DELAY(self, decoy): # NOTE: I implemented the delay to give a flat reward for every step that the red agent attacked a decoy.
        return 40.0 if decoy else 0
    
    def _DOWNTIME(self, blue_action, headstart, post_play, exceeded_decoy_limit, blue_success):
        if blue_success:
            if exceeded_decoy_limit:
                b = self.blue_rewards[blue_action][0] * 3
            elif headstart:
                b = (self.blue_rewards[blue_action][0] * -3) if (blue_action == "deploy_decoy") else self.blue_rewards[blue_action][0] #changed
            else: # after headstart
                if post_play: # after headstart actions are allowed
                    b = self.blue_rewards[blue_action][0]
                else:
                    b = self.blue_rewards[blue_action][0] * 60
        else:
            b = 0

        return b
    def _DETECT(self, blue_action, headstart, post_play, exceeded_decoy_limit, blue_success):
        if blue_success:
            if exceeded_decoy_limit:
                b = self.blue_rewards[blue_action][0] * 3
            elif headstart:
                b = (self.blue_rewards[blue_action][0] * -3) if (blue_action == "deploy_decoy") else self.blue_rewards[blue_action][0] #changed
            else: # after headstart
                if post_play: # after headstart actions are allowed
                    b = self.blue_rewards[blue_action][0]
                else:
                    b = self.blue_rewards[blue_action][0] * 60
        else:
            b = 0

        return b
    def _GENERAL(self, blue_action, headstart, post_play, exceeded_decoy_limit, blue_success):
        if blue_success:
            if exceeded_decoy_limit:
                b = self.blue_rewards[blue_action][0] * 3
            elif headstart:
                b = (self.blue_rewards[blue_action][0] * -3) if (blue_action == "deploy_decoy") else self.blue_rewards[blue_action][0] #changed
            else: # after headstart
                if post_play: # after headstart actions are allowed
                    b = self.blue_rewards[blue_action][0]
                else:
                    b = self.blue_rewards[blue_action][0] * 60
        else:
            b = 0

        return b

    def calculate_reward(
        self,
        red_action: str,
        blue_action: str,
        red_success: str,
        blue_success: bool,
        target_host: Host,
        blue_id: str = -1,
        blue_recurring: int = 0,
    ) -> int | float:
        exceeded_decoy_limit = self.network.get_num_decoys() >= self.args.decoy_limit
        objective = self.args.objective

        target_host_name = target_host.name
        decoy = target_host.decoy

        multiplier = 1

        if blue_action == "deploy_decoy" and exceeded_decoy_limit:
            multiplier = 3
        
        #multiplier = 3 if blue_action == "deploy_decoy" and exceeded_decoy_limit else -1 if blue_action == "deploy_decoy" and blue_success else 1

        b = self.blue_rewards[blue_action][0] * multiplier
        r = 0

        action_dict = {
            "delay": self._DELAY,
            "downtime": self._DOWNTIME,
            "detect": self._DETECT,
            "general": self._GENERAL
        }

        if objective in action_dict:
            b += action_dict[objective](decoy)


        if blue_recurring == -1:
            self.remove_recurring_blue_action(blue_id)
        elif blue_recurring == 1:
            self.add_recurring_blue_action(blue_id, blue_action)

        return r + b + self.sum_recurring()
    
    def sum_recurring(self) -> int | float:
        sum = 0
        for ra in self.blue_recurring_actions:
            sum += self.blue_rewards[ra.action][1]
        for ra in self.red_recurring_actions:
            sum += self.red_rewards[ra[0].action][1]
        return sum

    def add_recurring_blue_action(self, id: str, action: str) -> None:
        self.blue_recurring_actions.append(RecurringAction(id, action))

    def remove_recurring_blue_action(self, id: str) -> None:
        for i in range(len(self.blue_recurring_actions)):
            if self.blue_recurring_actions[i].id == id:
                self.blue_recurring_actions.pop(i)
                break

    def add_recurring_red_action(self, id: str, red_action: str, is_decoy: bool) -> None:
        self.red_recurring_actions.append((RecurringAction(id, red_action), is_decoy))

    def reset(self) -> None:
        self.blue_recurring_actions = []
        self.red_recurring_actions = []
