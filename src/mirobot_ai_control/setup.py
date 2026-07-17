from setuptools import find_packages, setup

package_name = 'mirobot_ai_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usr',
    maintainer_email='usr@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_node = mirobot_ai_control.robot_node:main',
            'ai_node = mirobot_ai_control.ai_node:main',
            'ai_node_new = mirobot_ai_control.ai_node_new:main',
            'gui_node = mirobot_ai_control.gui_node:main',
        ],
    },
)
