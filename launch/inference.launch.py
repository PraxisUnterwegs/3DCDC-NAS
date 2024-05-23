from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess 
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('AutoGesture')
    return LaunchDescription([
        Node(
            package='AutoGesture',
            executable='gesture_recognition_inference_node',
            name='gesture_recognition_inference_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                "model_config": os.path.join(pkg, "Multi_modality", "config.yml"),
                "rgb_topic": "/rgb/image_raw",
                "depth_topic": "/depth/image_raw",
            }]
        ),
    ])

