from setuptools import find_packages, setup

package_name = 'mirobot_ai_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # .npz(GRU 가중치)는 파이썬 파일이 아니어서 자동 복사되지 않는다.
    # 이 설정이 없으면 빌드 후 install 경로에 모델이 없어
    # 보정이 조용히 비활성화된다.
    package_data={package_name: ['*.npz']},
    include_package_data=True,
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
            'ai_node_new = mirobot_ai_control.ai_node_new:main',
            'robot_cam_node = mirobot_ai_control.robot_cam_node:main',
            'logger_node = mirobot_ai_control.ai_node_new_test_logger:main',
        ],
    },
)
