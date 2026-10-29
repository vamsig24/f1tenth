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
# the CommonRoad vehicle models). Tyre forces are mu * C_S * F_z * slip: the model's
# tyre_mu sets that cornering stiffness; the friction parameter sets the grip limit
# (how hard the planner may corner and brake). The gym uses one value for both.
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
    cmd_steer_rate  fastest the planner's steering command may change (rad/s)
    cmd_accel, cmd_decel  fastest its speed command may rise / fall (m/s per s)
    tyre_mu     the mu in the tyre model's cornering stiffness (mu * C_S)
    latency     time from the pose the controller plans from to its command acting
                on the car, beyond what the node measures (the pose's age and the
                solve time). The controller predicts the state that far ahead.
    """
    max_steer: float
    steer_rate: float
    kp_up: float
    kp_down: float
    headroom: float
    cmd_steer_rate: float
    cmd_accel: float
    cmd_decel: float
    tyre_mu: float
    latency: float

    def array(self):
        return np.array(self[:9], np.float32)


# f1tenth_gym: steering bang-bang at sv_max, P speed loop with kp = 10 a_max / v_max
# up and 10 a_max / |v_min| down (dynamic_models.pid), no integral so commands may
# exceed max_speed to accelerate; the drive command lands on the next 10 ms step.
SIM = Model(max_steer=0.4189, steer_rate=3.2, kp_up=10.0 * A_MAX / 20.0, kp_down=10.0 * A_MAX / 5.0,
            headroom=2.0, cmd_steer_rate=100.0, cmd_accel=1000.0, cmd_decel=1000.0, tyre_mu=1.0489, latency=0.01)

# Our car: /drive steering is clamped at 0.34 rad (the servo's end stops); the VESC
# holds the commanded speed, so no headroom. latency, kp_down and tyre_mu are fitted
# with scripts/model_check.py --fit on five bags (2026-10-25 x3, 10-27, 10-28): the
# median position error of the model's 0.5 and 1 s predictions fell from 20 to 16 cm,
# and each bag fitted without the others agreed (latency 0.08-0.10 s, kp_down 8,
# tyre_mu 1.0). The steering rate (2.5 rad/s) and kp_up (2) were left as measured
# before, the fits disagreed on them. Refit with new bags before changing these.
# Steering commands change at most at the servo's rated 3.2 rad/s less a margin; the
# planner's candidates change the speed command at most cmd_accel per second either
# way, and only its hard-stop candidate may cut it at cmd_decel. These are smoothing
# limits on the commands, not the car's acceleration (the kp gains model that).
CAR = Model(max_steer=0.34, steer_rate=2.5, kp_up=2.0, kp_down=8.0, headroom=0.0,
            cmd_steer_rate=2.8, cmd_accel=7.0, cmd_decel=20.0, tyre_mu=1.0, latency=0.10)


# Grades for the prediction error (median, cm): (horizon s, good below, okay below).
# The fitted CAR model replaying the 10-25..10-28 bags scores ~6 cm at 0.25 s,
# 11 at 0.5 s and 20 at 1.0 s; the latency prediction (~0.1 s) uses the first row.
PRED_GRADES = ((0.25, 8.0, 15.0), (0.5, 15.0, 30.0), (1.0, 30.0, 60.0))
PRED_BIAS = 0.5   # a mean along/cross offset above this share of the median error is systematic


def grade(horizon, err_cm, along_cm, cross_cm):
    """'GOOD' / 'OKAY' / 'BAD' for a median error, plus which way a systematic offset points."""
    _, good, okay = next((g for g in PRED_GRADES if horizon <= g[0]), PRED_GRADES[-1])
    status = 'GOOD' if err_cm < good else 'OKAY' if err_cm < okay else 'BAD'
    limit = max(PRED_BIAS * err_cm, 0.5 * good)
    hints = []
    if abs(along_cm) > limit:
        hints.append('model too fast' if along_cm > 0 else 'model too slow')
    if abs(cross_cm) > limit:
        hints.append('model drifts left' if cross_cm > 0 else 'model drifts right')
    return status, hints


def worst(statuses):
    return next((s for s in ('BAD', 'OKAY', 'GOOD') if s in statuses), 'GOOD')
