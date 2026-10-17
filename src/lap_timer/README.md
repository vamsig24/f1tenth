# AD Homework: Improving Pure Pursuit

In Lab 4 you wrote a pure pursuit node that follows waypoints around Levine at a constant speed. In this problem you time that controller over one lap, then modify it to lap faster without degrading tracking performance. The problem statement and the target lap time are in the homework PDF. This ROS 2 package contains the tools you need to time and plot your laps.

```
lap_timer/
├── lap_timer/
│   ├── lap_timer_node.py  # times a lap, checks it, saves a log
│   └── plot_lap.py        # plots the more lap logs
├── waypoints/levine.csv   # the waypoints you must follow
└── maps/                  # copy of the Levine map (used for plots)
```

## 1. Setup

You need your Lab 4 `pure_pursuit` package and this `lap_timer` package in the simulator workspace. If you use Docker, add both to the `volumes` of the `sim` service in `docker-compose.yml`, as in the previous labs:

```yaml
- <path_to>/lap_timer:/sim_ws/src/lap_timer
- <path_to_your_lab4>/pure_pursuit:/sim_ws/src/pure_pursuit
```

Then, inside the container:

```bash
cd /sim_ws
colcon build --packages-select lap_timer pure_pursuit
source install/setup.bash
```

The plotting script needs matplotlib, which the simulator image does not ship with. Install it inside the container with

```bash
pip3 install "matplotlib<3.9" "numpy==1.22.0"
```

Do not run a plain `pip3 install matplotlib`: it upgrades numpy to a version the simulator does not support.. This must be repeated if the container is recreated. Alternatively, run the script on your host machine (it needs `numpy`, `matplotlib` and `pillow`).

## 2. Rules

Every lap you report must use the following setup:

| | |
| --- | --- |
| Map | `levine`, unchanged `sim.yaml` |
| Start | standing start from the default spawn pose (0, 0, 0) |
| Waypoints | `lap_timer/waypoints/levine.csv`, unmodified |
| Wheelbase | $L_{wb}$ = 0.3302 m |
| Steering limit | $\lvert\delta\rvert \le$ 0.4 rad |
| Tracking error | at most 0.5 m from the waypoint path at all times |
| Collisions | none |

You may change the lookahead distance, the speed, and how either of them is chosen while driving.

Your **vanilla** controller is your Lab 4 node with a constant speed and the steering law $\delta = \arctan(L_{wb}\,\gamma)$, with $\gamma = 2y/L^2$ as in Lab 4. Tune its lookahead and speed for the fastest valid lap you can get. Your **augmented** controller is the modified node.

## 3. Timing a lap

Use three terminals in the container (`source /sim_ws/install/setup.bash` in each):

```bash
# 1. simulator
ros2 launch f1tenth_gym_ros gym_bridge_launch.py

# 2. lap timer (start it before your controller; press Enter here to reset the car)
ros2 run lap_timer lap_timer --ros-args -p tag:=vanilla -p log_dir:=/sim_ws/src/lap_timer/logs

# 3. your controller (make it load /sim_ws/src/lap_timer/waypoints/levine.csv)
ros2 run pure_pursuit pure_pursuit_node
```

The timer starts when the car starts moving and stops when it has driven one full loop of the waypoints. It then prints a line:

```
Lap: 11.99 s | max tracking error 0.451 m, mean 0.085 m | VALID
```

and saves the lap as `logs/<tag>_<date>-<time>_<status>.csv`, where the status is `valid`, `invalid` (tracking error above 0.5 m) or `dnf` (the car stopped, usually after a crash). With the `log_dir` above, the logs appear on your host in `lap_timer/logs/`.

To time another lap, press **Enter** in the lap timer terminal while your controller is running. This puts the car back at the spawn pose and restarts the timer. Do not use 2D Pose Estimate to reset the car, since it cannot place the car exactly at the spawn pose.

## 4. Plotting

```bash
cd /sim_ws/src/lap_timer
ros2 run lap_timer plot_lap logs/vanilla_*.csv logs/augmented_*.csv --labels vanilla augmented --out laps.png
```

or, on the host from the `lap_timer` folder, `python3 lap_timer/plot_lap.py ...` with the same arguments. The figure shows, for each lap, the path the car drove next to the waypoints on the map, and the tracking error of every lap against time with the 0.5 m limit.
