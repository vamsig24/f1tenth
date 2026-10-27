"""F1TENTH vehicle constants.

Geometry, steering and motor limits are the f1tenth_gym defaults (f110_env.py),
which come from the F1TENTH car (Traxxas Slash 4x4 chassis, VESC). The planner
uses a kinematic bicycle with the steady-state understeer of the gym's
single-track model, so nothing here needs tuning per deployment except the
friction coefficient, which is a launch parameter.
"""
import math

G = 9.81

LF = 0.15875              # CG to front axle (m)
LR = 0.17145              # CG to rear axle (m)
WHEELBASE = LF + LR
LENGTH = 0.58             # body (m)
WIDTH = 0.31
MAX_STEER = 0.4189        # rad
MAX_STEER_RATE = 3.2      # rad/s
A_MAX = 9.51              # m/s^2, also the braking limit
V_SWITCH = 7.319          # m/s, above this the motor limits accel to A_MAX * V_SWITCH / v

# Single-track tyre/inertia parameters (F1TENTH values used by f1tenth_gym and
# the CommonRoad vehicle models). Tyre forces are mu * C_S * F_z * slip, so the
# friction coefficient scales cornering stiffness as well as the grip limit.
C_SF = 4.718              # front cornering stiffness coefficient (1/rad)
C_SR = 5.4562             # rear
H_CG = 0.074              # CG height (m)
MASS = 3.74               # kg
INERTIA = 0.04712         # yaw inertia (kg m^2)
STEER_DELAY = 0.02        # s: the gym holds steering commands for two 10 ms steps

# Command latency: time from the pose the controller plans from to its command
# acting on the car. The node measures the pose's age and the solve time and
# adds these constants for the part it cannot see (transport and the
# servo/VESC response; one 10 ms step in the gym). The controller predicts the
# state that far ahead before planning. Guessing too high hurts more than
# too low, so they are kept small.
LATENCY_SIM = 0.01
LATENCY_CAR = 0.03

# Low-level speed loop. In the gym, /drive speed goes through a P controller
# (f110_gym dynamic_models.pid): accel = kp * (v_cmd - v), kp = 10 * a_max / v_max
# when speeding up and 10 * a_max / |v_min| when slowing down. On the car the
# VESC holds speed much more tightly; a stiff first-order loop models it.
SPEED_KP_SIM = (10.0 * A_MAX / 20.0, 10.0 * A_MAX / 5.0)
SPEED_KP_CAR = (20.0, 20.0)

# LiDAR mount, ahead of base_link (ego_racecar.xacro; same on the car).
LASER_X = 0.275

# base_link is the rear axle; the body runs from ~0.10 m behind it to ~0.48 m
# ahead (front bumper ~0.15 m past the front axle). This also covers the gym,
# whose collision check is a box around the laser reaching 0.44 m ahead.
BODY_CENTER_X = 0.19
# Footprint as circles along the body axis; five keep the rounding past the
# body sides to ~1 cm (three waste ~3 cm a side, which closes real gaps).
N_CIRCLES = 5
CIRCLE_OFFSETS = tuple(BODY_CENTER_X + (i - (N_CIRCLES - 1) / 2) * LENGTH / N_CIRCLES for i in range(N_CIRCLES))
CIRCLE_RADIUS = math.hypot(LENGTH / (2 * N_CIRCLES), WIDTH / 2)
