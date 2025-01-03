import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    preprocess_pkg_path = os.path.join(
        os.getenv('COLCON_PREFIX_PATH', '/'), 'share', 'preprocess_pkg'
    )
    image_processor_script = os.path.join(preprocess_pkg_path, 'scripts', 'image_processor.py')

    return LaunchDescription([
        # Launch image_processor.py from preprocess_pkg
        Node(
            package='preprocess_pkg',
            executable='image_processor.py',
            name='image_processor',
            output='screen',
            emulate_tty=True,  # Ensures output is printed cleanly
        ),
        # Launch reactive_node from driver_pkg
        Node(
            package='driver_pkg',
            executable='reactive_node',
            name='reactive_node',
            output='screen',
            emulate_tty=True,
        ),
        # Launch recognition_node.py from recognition_pkg
        Node(
            package='recognition_pkg',
            executable='recognition_node.py',
            name='recognition_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
