# Copyright (c) 2022 Hamburg Bit-Bots
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

from ipm_library.ipm import IPM
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from soccer_ipm.msgs.ball import map_ball_array
from soccer_ipm.msgs.field_boundary import map_field_boundary
from soccer_ipm.msgs.goalpost import map_goalpost_array
from soccer_ipm.msgs.markings import map_marking_array
from soccer_ipm.msgs.mask import map_masks
from soccer_ipm.msgs.obstacles import map_obstacle_array
from soccer_ipm.msgs.robots import map_robot_array
from soccer_ipm.utils import compose
import soccer_vision_2d_msgs.msg as sv2dm
import soccer_vision_3d_msgs.msg as sv3dm
import tf2_ros as tf2


class SoccerIPM(Node):

    def __init__(self) -> None:
        super().__init__('soccer_ipm')
        # We need to create a tf buffer
        self.tf_buffer = tf2.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self)

        # Create an IPM instance
        self.ipm = IPM(self.tf_buffer)

        # Declare params
        self.declare_parameter('balls.ball_diameter', 0.153)
        self.declare_parameter('output_frame', 'base_footprint')
        self.declare_parameter('goalposts.footpoint_out_of_image_threshold', 0.8)
        self.declare_parameter('obstacles.footpoint_out_of_image_threshold', 0.8)
        self.declare_parameter('robots.footpoint_out_of_image_threshold', 0.8)
        self.declare_parameter('masks.line_mask.scale', 0.0)

        # Subscribe to camera info
        self.create_subscription(CameraInfo, 'camera_info', self.ipm.set_camera_info, 1)

        # Create processing pipelines

        # Balls
        self.create_subscription(
            sv2dm.BallArray,
            'balls_in_image',
            compose(
                self.create_publisher(sv3dm.BallArray, 'balls_relative', 1).publish,
                partial(
                    map_ball_array,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger(),
                    ball_diameter=self.get_parameter('balls.ball_diameter').value),
                ),
            1)

        # Goal posts
        self.create_subscription(
            sv2dm.GoalpostArray,
            'goal_posts_in_image',
            compose(
                self.create_publisher(sv3dm.GoalpostArray, 'goal_posts_relative', 1).publish,
                partial(
                    map_goalpost_array,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger(),
                    footpoint_out_of_image_threshold=self.get_parameter(
                        'obstacles.footpoint_out_of_image_threshold').value),
                ),
            1)

        # Robots
        self.create_subscription(
            sv2dm.RobotArray,
            'robots_in_image',
            compose(
                self.create_publisher(sv3dm.RobotArray, 'robots_relative', 1).publish,
                partial(
                    map_robot_array,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger(),
                    footpoint_out_of_image_threshold=self.get_parameter(
                        'robots.footpoint_out_of_image_threshold').value),
                ),
            1)

        # Obstacles
        self.create_subscription(
            sv2dm.ObstacleArray,
            'obstalces_in_image',
            compose(
                self.create_publisher(sv3dm.ObstacleArray, 'obstacles_relative', 1).publish,
                partial(
                    map_obstacle_array,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger(),
                    footpoint_out_of_image_threshold=self.get_parameter(
                        'obstacles.footpoint_out_of_image_threshold').value),
                ),
            1)

        # Field boundary
        self.create_subscription(
            sv2dm.FieldBoundary,
            'field_boundary_in_image',
            compose(
                self.create_publisher(sv3dm.FieldBoundary, 'field_boundary_relative', 1).publish,
                partial(
                    map_field_boundary,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger()),
                ),
            1)

        # Markings
        self.create_subscription(
            sv2dm.MarkingArray,
            'markings_in_image',
            compose(
                self.create_publisher(sv3dm.MarkingArray, 'markings_relative', 1).publish,
                partial(
                    map_marking_array,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger()),
                ),
            1)

        # Masks
        self.create_subscription(
            Image,
            'line_mask_in_image',
            compose(
                self.create_publisher(PointCloud2, 'line_mask_relative_pc', 1).publish,
                partial(
                    map_masks,
                    ipm=self.ipm,
                    output_frame=self.get_parameter('output_frame').value,
                    logger=self.get_logger(),
                    scale=self.get_parameter('masks.line_mask.scale').value),
                ),
            1)


def main(args=None):
    rclpy.init(args=args)
    node = SoccerIPM()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    ex.spin()
    node.destroy_node()
    rclpy.shutdown()
