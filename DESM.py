import numpy as np
import scipy as sp
import numbers

mu = 398600.45e+09  # Гравитационный параметр Земли
R_e = 6371.0e+03  # Радиус Земли
J2 = 1.08262668e-3 # Коэффициент J2 для Земли, безразмерная величина, характеризующая отклонение от сферической формы
J3 = -2.5327e-6

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

def _J2_ODES(t, state_vector):
    x, y, z, vx, vy, vz = state_vector
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dxdt = vx
    dydt = vy
    dzdt = vz

    factorJ2 = (3 / 2) * J2 * mu * (R_e ** 2) / (r ** 5)
    z2_r2 = (z ** 2) / (r ** 2)

    dvxdt = -(mu * x) / (r ** 3) + factorJ2 * x * (5 * z2_r2 - 1)
    dvydt = -(mu * y) / (r ** 3) + factorJ2 * y * (5 * z2_r2 - 1)
    dvzdt = -(mu * z) / (r ** 3) + factorJ2 * z * (5 * z2_r2 - 3)

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

def _J3_ODES(t, state_vector):
    x, y, z, vx, vy, vz = state_vector
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dxdt = vx
    dydt = vy
    dzdt = vz

    factorJ2 = (3 / 2) * J2 * mu * (R_e ** 2) / (r ** 5)
    z2_r2 = (z ** 2) / (r ** 2)
    factorJ3 = (5 / 2) * J3 * mu * (R_e ** 3) / (r ** 7)
    z4_r2 = (z ** 4) / (r ** 2)

    dvxdt = -(mu * x) / (r ** 3) + factorJ2 * x * (5 * z2_r2 - 1) + factorJ3 * (x * z * (7 * z2_r2 - 3) )
    dvydt = -(mu * y) / (r ** 3) + factorJ2 * y * (5 * z2_r2 - 1) + factorJ3 * (y * z * (7 * z2_r2 - 3) )
    dvzdt = -(mu * z) / (r ** 3) + factorJ2 * z * (5 * z2_r2 - 3) + factorJ3 * (6 * (z ** 2) - (3 / 2) * (r ** 2) - 7 * z4_r2 )

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])



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


def get_solution(position: list, time_interval: list, speed_proj: list, amendments: str = 'None',) -> list:
    position = _validate_vector("position", position)
    speed_proj = _validate_vector("speed_proj", speed_proj)
    t0, t1 = _validate_time_interval(time_interval)

    if amendments == 'None':
        sol = sp.integrate.solve_ivp(_ODES, (t0, t1), position + speed_proj, method='DOP853', rtol=1e-9, atol=1e-9)
        x, y, z, vx, vy, vz = sol.y
        t = sol.t
        return [x, y, z, vx, vy, vz, t]
    elif amendments == 'J2':
        sol = sp.integrate.solve_ivp(_J2_ODES, (t0, t1), position + speed_proj, method='DOP853', rtol=1e-9, atol=1e-9)
        x, y, z, vx, vy, vz = sol.y
        t = sol.t
        return [x, y, z, vx, vy, vz, t]
    elif amendments == 'J3':
        sol = sp.integrate.solve_ivp(_J3_ODES, (t0, t1), position + speed_proj, method='DOP853', rtol=1e-9, atol=1e-9)
        x, y, z, vx, vy, vz = sol.y
        t = sol.t
        return [x, y, z, vx, vy, vz, t]






