from glob import glob

from setuptools import setup

package_name = 'scan_mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/waypoints', glob('waypoints/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vamsi',
    maintainer_email='vamsig@andrew.cmu.edu',
    description='MPPI that tracks a plain (x, y) reference line and avoids LiDAR returns.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'scan_mppi_node = scan_mppi.node:main',
        ],
    },
)
