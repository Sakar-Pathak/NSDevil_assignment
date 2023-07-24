from setuptools import setup

package_name = 'unicycle_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sakar',
    maintainer_email='sakar.pathak111@gmail.com',
    description='NSDevil Hiring Assignment Solution',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'unicycle_robot = unicycle_robot.robot_kine_function:main',
            'unicycle_robot_simulator = unicycle_robot.robot_simulator:main',
        ],
    },
)
