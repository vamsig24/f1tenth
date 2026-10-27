"""F1TENTH vehicle constants.

Geometry, tyre and motor parameters are the f1tenth_gym defaults (f110_env.py),
which come from the F1TENTH car (Traxxas Slash 4x4 chassis, VESC). What differs
between the gym and our car (steering limit and servo speed, the VESC speed
loop, latency) is in the two Models at the bottom; the car's were measured
from its 2026-10-25 bags. Nothing here needs tuning per deployment except the
friction coefficient, which is a launch parameter.
"""
import math
from typing import NamedTuple

import numpy as np

G = 9.81

LF = 0.15875              # CG to front axle (m)
LR = 0.17145              # CG to rear axle (m)
WHEELBASE = LF + LR
LENGTH = 0.58             # body (m)
WIDTH = 0.31
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

# LiDAR mount, ahead of base_link (ego_racecar.xacro; 0.27 on the car's tf).
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


class Model(NamedTuple):
    """What the planner needs to know about the low-level control of one vehicle.

    max_steer   steering command limit (rad)
    steer_rate  steering slew rate (rad/s)
    kp_up/down  speed loop: accel = kp * (v_cmd - v), speeding up / slowing down (1/s)
    headroom    how far above max_speed a speed command may go (m/s)
    latency     time from the pose the controller plans from to its command acting
                on the car, beyond what the node measures (the pose's age and the
                solve time). The controller predicts the state that far ahead.
    """
    max_steer: float
    steer_rate: float
    kp_up: float
    kp_down: float
    headroom: float
    latency: float

    def array(self):
        return np.array(self[:5], np.float32)


# f1tenth_gym: steering bang-bang at sv_max, P speed loop with kp = 10 a_max / v_max
# up and 10 a_max / |v_min| down (dynamic_models.pid), no integral so commands may
# exceed max_speed to accelerate; the drive command lands on the next 10 ms step.
SIM = Model(max_steer=0.4189, steer_rate=3.2, kp_up=10.0 * A_MAX / 20.0, kp_down=10.0 * A_MAX / 5.0,
            headroom=2.0, latency=0.01)

# Our car, fitted to the 2026-10-25 bags (attem2, attem3): /drive steering is
# clamped at 0.34 rad (the servo's end stops); the yaw response matches a 2.5 rad/s
# slew after ~0.14 s of delay plus ~0.03 s of particle-filter lag; the VESC speed loop
# behaves like kp 2 up / 4 down. The VESC holds the commanded speed, so no headroom.
CAR = Model(max_steer=0.34, steer_rate=2.5, kp_up=2.0, kp_down=4.0, headroom=0.0, latency=0.15)
