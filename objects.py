import numpy as np

class Engine:
    def __init__(self, mass, Isp, max_thrust):
        self.mass = mass
        self.Isp = Isp
        self.max_thrust = max_thrust

class Satellite:
    def __init__(self, id, dry_mass, prop_mass, engine = None):
        self.id = id
        self.mass = dry_mass + prop_mass
        self.prop_mass = prop_mass
        self.engine = engine

class Maneuver:
    def __init__(self, satellite: Satellite, duration, xthrust, ythrust, zthrust):
        self.sat = satellite
        self.duration = duration
        if np.sqrt(xthrust**2 + ythrust**2 + zthrust**2) > self.sat.engine.max_thrust:
            raise ValueError("Thrust exceeds engine's maximum thrust, unable to create maneuver")
        else:
            self.xthrust = xthrust
            self.ythrust = ythrust
            self.zthrust = zthrust

class Trajectory:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.vx = []
        self.vy = []
        self.vz = []
