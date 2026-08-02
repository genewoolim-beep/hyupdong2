from setuptools import setup
import glob

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob.glob('resource/*')),
        ('share/' + package_name + '/resource', glob.glob('resource/.env')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey4090',
    maintainer_email='rokey4090@todo.todo',
    description='robot_control node',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'robot_control = robot_control.robot_control:main',
            'person_tracker = robot_control.person_tracker:main',
        ],
    },
)
