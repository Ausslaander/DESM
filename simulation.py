from objects import Maneuver, Satellite
from ODESM_engine import Propagator

class MissionSimulator:
    def __init__(self, start_epoch, satellite):
        self.start_epoch = start_epoch
        self.satellite = satellite
        self.propagator = Propagator()