"""ros2 launch scan_mppi scan_mppi.launch.py [config:=sim|car] [waypoints_file:=...] [max_speed:=...] [friction:=...]

config picks config/<config>.yaml; the other arguments override it.

With config:=sim and no waypoints_file, the reference follows the gym's map:
map_path from f1tenth_gym_ros's sim.yaml is looked up in config/tracks.yaml.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node


def gym_map():
    """Name of the map the gym bridge will load, or None."""
    try:
        path = os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'config', 'sim.yaml')
        with open(path) as f:
            return os.path.basename(yaml.safe_load(f)['bridge']['ros__parameters']['map_path'])
    except Exception:
        return None


def track_reference(share, map_name):
    with open(os.path.join(share, 'config', 'tracks.yaml')) as f:
        tracks = yaml.safe_load(f) or {}
    match = max((k for k in tracks if map_name.startswith(k)), key=len, default=None)
    return tracks[match] if match else None


def setup(context):
    cfg = context.launch_configurations
    share = get_package_share_directory('scan_mppi')
    params = [os.path.join(share, 'config', cfg['config'] + '.yaml')]
    overrides = {k: cfg[k] for k in ('waypoints_file',) if cfg.get(k)}
    for k in ('max_speed', 'friction'):
        if cfg.get(k):
            overrides[k] = float(cfg[k])
    actions = []
    if cfg['config'] == 'sim' and 'waypoints_file' not in overrides:
        map_name = gym_map()
        ref = track_reference(share, map_name) if map_name else None
        if ref:
            overrides['waypoints_file'] = ref
            actions.append(LogInfo(msg=f'scan_mppi: gym map {map_name} -> reference {ref}'))
        else:
            actions.append(LogInfo(msg=f'scan_mppi: no tracks.yaml entry for gym map {map_name}; '
                                       f'using waypoints_file from {cfg["config"]}.yaml'))
    if overrides:
        params.append(overrides)
    env = {}
    if cfg['config'] == 'car' and os.path.isdir('/usr/local/cuda/lib64'):
        # JetPack's CUDA libraries for GPU JAX (scripts/setup_jetson.sh)
        env['LD_LIBRARY_PATH'] = ':'.join(p for p in ('/usr/local/cuda/lib64', os.environ.get('LD_LIBRARY_PATH')) if p)
    return actions + [Node(package='scan_mppi', executable='scan_mppi_node', name='scan_mppi',
                           output='screen', parameters=params, additional_env=env)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='sim'),
        DeclareLaunchArgument('waypoints_file', default_value=''),
        DeclareLaunchArgument('max_speed', default_value=''),
        DeclareLaunchArgument('friction', default_value=''),
        OpaqueFunction(function=setup),
    ])
