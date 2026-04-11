import matplotlib.pyplot as plt
import numpy as np

from DESM import R_e, get_solution, mu


def compute_orbit(altitude_m: float, duration_s: float):
	radius = R_e + altitude_m
	position = [radius, 0.0, 0.0]
	speed_proj = [0.0, np.sqrt(mu / radius), 0.0]
	x, y, z, vx, vy, vz, t = get_solution(position, [0.0, duration_s], speed_proj, use_J2=False)
	return np.asarray(x), np.asarray(y), np.asarray(z), np.asarray(vx), np.asarray(vy), np.asarray(vz), np.asarray(t)


def nearest_point_to_equator(x_values, y_values, z_values):
	idx = int(np.argmin(np.abs(z_values)))
	return x_values[idx], y_values[idx], z_values[idx]


def plot_orbit(x_values, y_values, z_values, point):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	# Earth sphere
	u = np.linspace(0.0, 2.0 * np.pi, 100)
	v = np.linspace(0.0, np.pi, 100)
	x = R_e * np.outer(np.cos(u), np.sin(v))
	y = R_e * np.outer(np.sin(u), np.sin(v))
	z = R_e * np.outer(np.ones(np.size(u)), np.cos(v))
	ax.plot_surface(x, y, z, color="b", alpha=0.3)

	# Orbit and marker
	ax.scatter(point[0], point[1], point[2], color="g", s=50, label="Point closest to Z=0")
	ax.plot(x_values, y_values, z_values, label="Satellite orbit", color="r")

	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.set_zlabel("Z (m)")

	# Equal limits on all axes so Earth remains visually spherical.
	limit = 1.05 * max(
		float(np.max(np.abs(x_values))),
		float(np.max(np.abs(y_values))),
		float(np.max(np.abs(z_values))),
		float(R_e),
	)
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)
	if hasattr(ax, "set_box_aspect"):
		ax.set_box_aspect((1, 1, 1))

	ax.set_title("Satellite orbit and Earth")
	ax.legend()
	plt.show()


def main():
    x = R_e+400*10**3
    y = 0
    z = 0
    vx = 0
    vy = np.sqrt(mu/(R_e+400e3)) + 1000
    vz = 0
    t = [0,7200]
    solution = get_solution([x,y,z], t, [vx,vy,vz], amendments='J3')
    x, y, z, vx, vy, vz, t = solution
    point = nearest_point_to_equator(x, y, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    print(max(r), min(r))
    plot_orbit(x, y, z, point)


if __name__ == "__main__":
	main()

