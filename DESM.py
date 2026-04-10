import numpy as np
import scipy as sp
import numbers

mu = 398600.45e+09  # Гравитационный параметр Земли
R_m = 6371.0e+03  # Радиус Земли

def _ODES(t, state_vector):
    x,y,z,vx,vy,vz = state_vector
    r = np.sqrt(x**2 + y**2 + z**2)
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -(mu*x)/(r**3)
    dvydt = -(mu*y)/(r**3)
    dvzdt = -(mu*z)/(r**3)
    return np.array([dxdt,dydt,dzdt, dvxdt,dvydt,dvzdt])


def _validate_vector(name: str, value) -> list:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} must be a sequence of 3 numeric values")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")

    validated = []
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise TypeError(f"{name}[{idx}] must be a real number")
        if not np.isfinite(item):
            raise ValueError(f"{name}[{idx}] must be finite")
        validated.append(float(item))
    return validated


def _validate_time_interval(time_interval) -> tuple:
    if not isinstance(time_interval, (list, tuple, np.ndarray)):
        raise TypeError("time_interval must be a sequence of 2 numeric values")
    if len(time_interval) != 2:
        raise ValueError("time_interval must contain exactly 2 values: [t0, t1]")

    t0, t1 = time_interval
    if isinstance(t0, bool) or not isinstance(t0, numbers.Real):
        raise TypeError("time_interval[0] must be a real number")
    if isinstance(t1, bool) or not isinstance(t1, numbers.Real):
        raise TypeError("time_interval[1] must be a real number")
    if not np.isfinite(t0) or not np.isfinite(t1):
        raise ValueError("time_interval values must be finite")
    if t1 <= t0:
        raise ValueError("time_interval must satisfy t1 > t0")

    return float(t0), float(t1)


def get_solution(position: list, time_interval: list, speed_proj: list, use_J2: bool = False) -> list:
    position = _validate_vector("position", position)
    speed_proj = _validate_vector("speed_proj", speed_proj)
    t0, t1 = _validate_time_interval(time_interval)

    if use_J2:
        pass
    else:
        sol = sp.integrate.solve_ivp(_ODES, (t0, t1), position + speed_proj, method = 'DOP853')
        x,y,z,vx,vy,vz = sol.y
        t = sol.t
        return [x,y,z,vx,vy,vz,t]




