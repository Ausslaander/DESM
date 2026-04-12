import numpy as np
import scipy as sp
import numbers

from config import mu, R_e, J2, J3, g0


def _ODES(time, state_vector):
    x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp = state_vector
    r = np.sqrt(x**2 + y**2 + z**2)

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -(mu*x)/(r**3) + Fx / m
    dvydt = -(mu*y)/(r**3) + Fy / m
    dvzdt = -(mu*z)/(r**3) + Fz / m

    dFx = 0.0
    dFy = 0.0
    dFz = 0.0

    dmdt = -np.sqrt(Fx**2 + Fy**2 + Fz**2) / (g0 * Isp)
    dIspdt = 0.0

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dFx, dFy, dFz, dmdt, dIspdt])

def _J2_ODES(time, state_vector):
    x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp = state_vector
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dxdt = vx
    dydt = vy
    dzdt = vz

    factorJ2 = (3 / 2) * J2 * mu * (R_e ** 2) / (r ** 5)
    z2_r2 = (z ** 2) / (r ** 2)

    dvxdt = -(mu * x) / (r ** 3) + factorJ2 * x * (5 * z2_r2 - 1) + Fx / m
    dvydt = -(mu * y) / (r ** 3) + factorJ2 * y * (5 * z2_r2 - 1) + Fy / m
    dvzdt = -(mu * z) / (r ** 3) + factorJ2 * z * (5 * z2_r2 - 3) + Fz / m

    dFxdt = 0.0
    dFydt = 0.0
    dFzdt = 0.0
    dmdt = -np.sqrt(Fx**2 + Fy**2 + Fz**2) / (g0 * Isp)
    dIspdt = 0.0

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dFxdt, dFydt, dFzdt, dmdt, dIspdt])

def _J3_ODES(time, state_vector):
    x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp = state_vector
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dxdt = vx
    dydt = vy
    dzdt = vz

    factorJ2 = (3 / 2) * J2 * mu * (R_e ** 2) / (r ** 5)
    z2_r2 = (z ** 2) / (r ** 2)
    factorJ3 = (5 / 2) * J3 * mu * (R_e ** 3) / (r ** 7)
    z4_r2 = (z ** 4) / (r ** 2)

    dvxdt = -(mu * x) / (r ** 3) + factorJ2 * x * (5 * z2_r2 - 1) + factorJ3 * (x * z * (7 * z2_r2 - 3)) + Fx / m
    dvydt = -(mu * y) / (r ** 3) + factorJ2 * y * (5 * z2_r2 - 1) + factorJ3 * (y * z * (7 * z2_r2 - 3)) + Fy / m
    dvzdt = -(mu * z) / (r ** 3) + factorJ2 * z * (5 * z2_r2 - 3) + factorJ3 * (6 * (z ** 2) - (3 / 2) * (r ** 2) - 7 * z4_r2) + Fz / m
    dFxdt = 0.0
    dFydt = 0.0
    dFzdt = 0.0
    dmdt = -np.sqrt(Fx**2 + Fy**2 + Fz**2) / (g0 * Isp)
    dIspdt = 0.0

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dFxdt, dFydt, dFzdt, dmdt, dIspdt])

def _validate_vector(vector):
    if not isinstance(vector, (list, tuple, np.ndarray)):
        raise TypeError("state_vector must be a sequence of 11 numeric values")

    if len(vector) != 11:
        raise ValueError(
            "state_vector must contain exactly 11 values: [x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp]"
        )

    validated = []
    for idx, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"state_vector[{idx}] must be a real number")
        if not np.isfinite(value):
            raise ValueError(f"state_vector[{idx}] must be finite")
        validated.append(float(value))

    if validated[9] <= 0.0:
        raise ValueError("state_vector[9] (m) must be > 0")
    if validated[10] <= 0.0:
        raise ValueError("state_vector[10] (Isp) must be > 0")

    return validated



class Propagator:
    def __init__(self, amendments: str = 'None', method: str = 'DOP853', rtol: float = 1e-9, atol: float = 1e-9):
        self.amendments = amendments
        self.method = method
        self.rtol = rtol
        self.atol = atol

    @staticmethod
    def _validate_duration(duration) -> float:
        if not isinstance(duration, (float, np.float64, int)):
            raise TypeError("duration must be float, np.float64 или int")
        if duration <= 0:
            raise ValueError("duration must be >0")

        t1 = duration
        if not np.isfinite(t1):
            raise ValueError("duration values must be finite")
        return float(t1)

    @staticmethod
    def _validate_dry_mass(dry_mass) -> float:
        if isinstance(dry_mass, bool) or not isinstance(dry_mass, numbers.Real):
            raise TypeError("dry_mass must be a real number")
        if not np.isfinite(dry_mass):
            raise ValueError("dry_mass must be finite")
        if dry_mass <= 0.0:
            raise ValueError("dry_mass must be > 0")
        return float(dry_mass)

    def _select_ode(self):
        if self.amendments == 'None':
            return _ODES
        if self.amendments == 'J2':
            return _J2_ODES
        if self.amendments == 'J3':
            return _J3_ODES
        raise ValueError("amendments must be one of: 'None', 'J2', 'J3'")

    def predict_trajectory(self, state_vector: list, duration: float, dry_mass: float) -> list:
        initial_state = _validate_vector(state_vector)
        t1 = self._validate_duration(duration)
        ode_fun = self._select_ode()


        dry_mass = self._validate_dry_mass(dry_mass)
        m0 = initial_state[9]
        if m0 <= dry_mass:
            raise ValueError("initial mass m must be greater than dry_mass")

        def mass_depletion_event(time, state):
            return state[9] - dry_mass

        mass_depletion_event.terminal = True
        mass_depletion_event.direction = -1

        # Phase 1: propagate with thrust until fuel reaches dry_mass or time ends.
        sol1 = sp.integrate.solve_ivp(
            ode_fun,
            (0.0, t1),
            initial_state,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            events=mass_depletion_event,
        )

        if not sol1.success:
            raise RuntimeError(f"Integration failed: {sol1.message}")

        fuel_depleted = sol1.t_events and len(sol1.t_events[0]) > 0
        if not fuel_depleted:
            x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp = sol1.y
            return [sol1.t, x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp]

        t_depletion = float(sol1.t_events[0][0])
        if t_depletion >= t1:
            x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp = sol1.y
            return [sol1.t, x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp]

        # Phase 2: continue to final time with thrust disabled and dry mass fixed.
        restart_state = sol1.y[:, -1].copy()
        restart_state[6] = 0.0
        restart_state[7] = 0.0
        restart_state[8] = 0.0
        restart_state[9] = dry_mass

        sol2 = sp.integrate.solve_ivp(
            ode_fun,
            (t_depletion, t1),
            restart_state,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not sol2.success:
            raise RuntimeError(f"Integration failed: {sol2.message}")

        t_full = np.concatenate((sol1.t, sol2.t[1:]))
        y_full = np.hstack((sol1.y, sol2.y[:, 1:]))
        x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp = y_full
        return [t_full, x, y, z, vx, vy, vz, Fx, Fy, Fz, m, Isp]






