import numpy as np

from typing import Dict, Iterable
from importlib.resources import files

from cyberwheel.detectors.alert import Alert
from cyberwheel.network.host import Host
from cyberwheel.observation.blue_observation import BlueObservation
from cyberwheel.detectors.handler import DetectorHandler

class BlueObservationProactive(BlueObservation):
    """
    This proactive blue agent deploys a set amount of decoys before the simulated cyber attack occurs.

    The observation space includes two more additional attributes:
    1. The number of decoys currently deployed.
    2. If the agent is in the headstart phase or not.

    We pass these additional attributes to BlueObservationProactive, where it appends to the end of the observation space.
    """

    def __init__(self, shape: int, mapping: Dict[Host, int], detector_config: str) -> None:
        super().__init__(shape, mapping, detector_config)

    def create_obs_vector(self, alerts: Iterable[Alert], headstart: bool, num_decoys: int) -> Iterable:
        # Refresh the non-history portion of the obs_vec

        if headstart:
            for i in range(self.len_obs):
                self.obs_vec[i] = 0
            self.obs_vec[-self.offset] = 1
            self.obs_vec[-self.offset + 1] = num_decoys
            return self.obs_vec

        barrier = self.len_obs // 2

        for i in range(barrier):
            self.obs_vec[i] = 0
        for alert in alerts:
            alerted_host = alert.src_host
            if not alerted_host or alerted_host.name not in self.mapping:
                continue
            index = self.mapping[alerted_host.name]
            self.obs_vec[index] = 1
            self.obs_vec[index + barrier] = 1
        self.obs_vec[-self.offset] = 0
        self.obs_vec[-self.offset + 1] = num_decoys # changed
        return self.obs_vec