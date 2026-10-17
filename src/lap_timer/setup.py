from glob import glob

from setuptools import setup

package_name = 'lap_timer'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/waypoints', glob('waypoints/*.csv')),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='16-665 TAs',
    description='Lap timer and lap plots for the 16-665 pure pursuit assignment',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lap_timer = lap_timer.lap_timer_node:main',
            'plot_lap = lap_timer.plot_lap:main',
        ],
    },
)
