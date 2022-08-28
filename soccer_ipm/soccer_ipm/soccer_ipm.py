import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3
from ipm_interfaces.msg import PlaneStamped, Point2DStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
import soccer_vision_2d_msgs.msg as sv2dm
import soccer_vision_3d_msgs.msg as sv3dm
from std_msgs.msg import Header
import tf2_ros as tf2
from vision_msgs.msg import BoundingBox2D, Point2D


class SoccerIPM(Node):

    def __init__(self) -> None:
        super().__init__('soccer_ipm')
        # We need to create a tf buffer
        self.tf_buffer = tf2.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self)

        # Create an IPM instance
        self.ipm = IPM(self.tf_buffer)

        # Create CvBride
        self._cv_bridge = CvBridge()

        # Declare params
        self.declare_parameter('balls.ball_diameter', 0.153)
        self.declare_parameter('output_frame', 'base_footprint')
        self.declare_parameter('obstacles.footpoint_out_of_image_threshold', 0.8)
        self.declare_parameter('goalposts.footpoint_out_of_image_threshold', 0.8)
        self.declare_parameter('masks.line_mask.scale', 0.0)

        # Parameters
        self._ball_diameter = self.get_parameter('balls.ball_diameter').value
        self._output_frame = self.get_parameter('output_frame').value
        self._obstacle_footpoint_out_of_image_threshold = \
            self.get_parameter('obstacles.footpoint_out_of_image_threshold').value
        self._goalpost_footpoint_out_of_image_threshold = \
            self.get_parameter('goalposts.footpoint_out_of_image_threshold').value
        line_mask_scaling = self.get_parameter('masks.line_mask.scale').value

        # Subscribe to camera info
        self.create_subscription(CameraInfo, 'camera_info', self.ipm.set_camera_info, 1)

        # Create publishers for 3d topics
        self.balls_pub = self.create_publisher(
            sv3dm.BallArray, 'balls_relative', 1)
        self.line_mask_pub = self.create_publisher(
            PointCloud2, 'line_mask_relative_pc', 1)
        self.goalposts_pub = self.create_publisher(
            sv3dm.GoalpostArray, 'goal_posts_relative', 1)
        self.robots_pub = self.create_publisher(
            sv3dm.RobotArray, 'robots_relative', 1)
        self.obstacles_pub = self.create_publisher(
            sv3dm.ObstacleArray, 'obstacles_relative', 1)
        self.field_boundary_pub = self.create_publisher(
            sv3dm.FieldBoundary, 'field_boundary_relative', 1)

        # Subscribe to image space data topics
        self.create_subscription(
            sv2dm.BallArray, 'balls_in_image', self.callback_ball, 1)
        self.create_subscription(
            sv2dm.GoalpostArray, 'goal_posts_in_image', self.callback_goalposts, 1)
        self.create_subscription(
            sv2dm.RobotArray, 'robots_in_image', self.callback_robots, 1)
        self.create_subscription(
            sv2dm.ObstacleArray, 'obstalces_in_image', self.callback_obstacles, 1)
        self.create_subscription(
            sv2dm.FieldBoundary, 'field_boundary_in_image', self.callback_field_boundary, 1)
        self.create_subscription(
            Image,
            'line_mask_in_image',
            lambda msg: self.callback_masks(
                msg,
                self.line_mask_pub,
                scale=line_mask_scaling), 1)

    def get_field(self, time, heigh_offset=0):
        plane = PlaneStamped()
        plane.header.frame_id = self._output_frame
        plane.header.stamp = time
        plane.plane.coef[2] = 1.0  # Normal in z direction
        plane.plane.coef[3] = -heigh_offset  # Distance above the ground
        return plane

    def callback_ball(self, msg: sv2dm.BallArray):
        field = self.get_field(msg.header.stamp, self._ball_diameter / 2)

        balls_relative = sv3dm.BallArray()
        balls_relative.header.stamp = msg.header.stamp
        balls_relative.header.frame_id = self._output_frame

        ball: sv2dm.Ball
        for ball in msg.balls:
            ball_point = Point2DStamped(
                header=msg.header,
                point=ball.center)
            try:
                transformed_ball = self.ipm.map_point(
                    field,
                    ball_point,
                    output_frame=self._output_frame)

                ball_relative = sv3dm.Ball()
                ball_relative.center = transformed_ball.point
                ball_relative.confidence = ball.confidence
                balls_relative.balls.append(ball_relative)
            except NoIntersectionError:
                self.get_logger().warn(
                    'Got a ball at ({},{}) I could not transform.'.format(
                        ball.center.x,
                        ball.center.y),
                    throttle_duration_sec=5)

        self.balls_pub.publish(balls_relative)

    def callback_goalposts(self, msg: sv2dm.GoalpostArray):
        field = self.get_field(msg.header.stamp)

        # Create new message
        goalposts_relative_msg = sv3dm.GoalpostArray()
        goalposts_relative_msg.header.stamp = msg.header.stamp
        goalposts_relative_msg.header.frame_id = self._output_frame

        # Transform goal posts
        goal_post_in_image: sv2dm.Goalpost
        for goal_post_in_image in msg.posts:
            # Check if post is not going out of the image at the bottom
            if not self._object_at_bottom_of_image(
                    self._bb_footpoint(goal_post_in_image.bb).y,
                    self._goalpost_footpoint_out_of_image_threshold):
                # Create footpoint
                footpoint = Point2DStamped(
                    header=msg.header,
                    point=self._bb_footpoint(goal_post_in_image.bb)
                )
                # Map point from image onto field plane
                try:
                    relative_foot_point = self.ipm.map_point(
                        field,
                        footpoint,
                        output_frame=self._output_frame)

                    post_relative = sv3dm.Goalpost()
                    post_relative.attributes = goal_post_in_image.attributes
                    post_relative.bb.center.position = relative_foot_point.point
                    post_relative.bb.size.x = 0.1  # TODO better size estimation
                    post_relative.bb.size.y = 0.1  # TODO better size estimation
                    post_relative.bb.size.z = 1.5  # TODO better size estimation
                    post_relative.confidence = goal_post_in_image.confidence
                    goalposts_relative_msg.posts.append(post_relative)
                except NoIntersectionError:
                    self.get_logger().warn(
                        'Got a post with foot point ({},{}) I could not transform.'.format(
                            footpoint.point.x,
                            footpoint.point.y),
                        throttle_duration_sec=5)

        self.goalposts_pub.publish(goalposts_relative_msg)

    def callback_robots(self, msg: sv2dm.RobotArray):
        field = self.get_field(msg.header.stamp, 0.0)

        robots = sv3dm.RobotArray()
        robots.header.stamp = msg.header.stamp
        robots.header.frame_id = self._output_frame

        robot: sv2dm.Robot
        for robot in msg.robots:

            # Check if post is not going out of the image at the bottom
            if not self._object_at_bottom_of_image(
                    self._bb_footpoint(robot.bb).y,
                    self._goalpost_footpoint_out_of_image_threshold):
                # Create footpoint
                footpoint = Point2DStamped(
                    header=msg.header,
                    point=self._bb_footpoint(robot.bb)
                )
                # Map point from image onto field plane
                try:
                    relative_foot_point = self.ipm.map_point(
                        field,
                        footpoint,
                        output_frame=self._output_frame)

                    transformed_robot = sv3dm.Robot()
                    transformed_robot.attributes = robot.attributes
                    transformed_robot.confidence = robot.confidence
                    transformed_robot.bb.center.position = relative_foot_point.point
                    transformed_robot.bb.size.x = 0.3   # TODO better size estimation
                    transformed_robot.bb.size.y = 0.3   # TODO better size estimation
                    transformed_robot.bb.size.z = 0.5   # TODO better size estimation
                    robots.robots.append(transformed_robot)
                except NoIntersectionError:
                    self.get_logger().warn(
                        'Got a robot with foot point ({},{}) I could not transform.'.format(
                            footpoint.point.x,
                            footpoint.point.y),
                        throttle_duration_sec=5)

        self.robots_pub.publish(robots)

    def callback_obstacles(self, msg: sv2dm.ObstacleArray):
        field = self.get_field(msg.header.stamp, 0.0)

        obstacles = sv3dm.ObstacleArray()
        obstacles.header.stamp = msg.header.stamp
        obstacles.header.frame_id = self._output_frame

        obstacle: sv2dm.Obstacle
        for obstacle in msg.obstacles:

            # Check if post is not going out of the image at the bottom
            if not self._object_at_bottom_of_image(
                    self._bb_footpoint(obstacle.bb).y,
                    self._goalpost_footpoint_out_of_image_threshold):
                # Create footpoint
                footpoint = Point2DStamped(
                    header=msg.header,
                    point=self._bb_footpoint(obstacle.bb)
                )
                # Map point from image onto field plane
                try:
                    relative_foot_point = self.ipm.map_point(
                        field,
                        footpoint,
                        output_frame=self._output_frame)
                    transformed_obstacle = sv3dm.Obstacle()
                    transformed_obstacle.confidence = obstacle.confidence
                    transformed_obstacle.bb.center.position = relative_foot_point.point
                    transformed_obstacle.bb.size.x = 0.3   # TODO better size estimation
                    transformed_obstacle.bb.size.y = 0.3   # TODO better size estimation
                    transformed_obstacle.bb.size.z = 0.5   # TODO better size estimation
                    obstacles.obstacles.append(transformed_obstacle)
                except NoIntersectionError:
                    self.get_logger().warn(
                        'Got a obstacle with foot point ({},{}) I could not transform.'.format(
                            footpoint.point.x,
                            footpoint.point.y),
                        throttle_duration_sec=5)

        self.obstacles_pub.publish(obstacles)

    def callback_markings(self, msg: sv2dm.MarkingArray):
        field = self.get_field(msg.header.stamp, 0.0)

        markings = sv3dm.MarkingArray()
        markings.header.stamp = msg.header.stamp
        markings.header.frame_id = self._output_frame

        ######################
        # Line intersections #
        ######################

        intersection: sv2dm.MarkingIntersection
        for intersection in msg.intersections:
                # Create center point
                center = Point2DStamped(
                    header=msg.header,
                    point=intersection.center
                )
                # Map point from image onto field plane
                try:
                    mapped_center_point = self.ipm.map_point(
                        field,
                        center,
                        output_frame=self._output_frame)
                    mapped_intersection = sv3dm.MarkingIntersection()
                    mapped_intersection.center = mapped_center_point.point
                    mapped_intersection.confidence = intersection.confidence
                    mapped_intersection.num_rays = intersection.num_rays

                    # Project rays
                    for ray in intersection.heading_rays:
                        # Create heading vector in image space
                        ray_vector = np.array([np.sin(ray), np.cos(ray)])
                        # Add heading vector to the center of the marking in image space
                        ray_end_point = center + ray_vector
                        # Map newly optained end point of the ray
                        mapped_ray_end = self.ipm.map_point(
                            field,
                            ray_end_point,
                            output_frame=self._output_frame)
                        # Substract the center point from the ray end point to get the vector
                        ray_mapped = Vector3(
                            x = mapped_ray_end.point.x - mapped_center_point.point.x,
                            y = mapped_ray_end.point.y - mapped_center_point.point.y,
                            z = mapped_ray_end.point.z - mapped_center_point.point.z
                        )
                        mapped_intersection.rays.append(ray_mapped)
                    markings.intersections.append(mapped_intersection)
                except NoIntersectionError:
                    self.get_logger().warn(
                        'Got a obstacle with foot point ({},{}) I could not transform.'.format(
                            mapped_center_point.point.x,
                            mapped_center_point.point.y),
                        throttle_duration_sec=5)

        #################
        # Line Segments #
        #################

        # Convert segment start and end points to np array
        segment_points_np = np.array([(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y)) for segment in msg.segments])

        # Map all points at once from image onto field plane
        segments_on_plane = self.ipm.map_points(
            field,
            segment_points_np.reshape(-1, 2),
            msg.header,
            output_frame=self._output_frame).reshape(-1, 2, 3)

        for i, segment in enumerate(segments_on_plane):
            # Check if any of the points failed to map
            if np.any(np.is_nan(segment)):
                self.get_logger().warn(
                    'Got a segment I could not transform.',
                    throttle_duration_sec=5)
                continue
            start, end = segment
            segment_msg = sv3dm.MarkingSegment()
            # Get confidence value from original list
            segment_msg.confidence.confidence = msg.segments[i].confidence.confidence
            # Convert np array -> Point
            segment_msg.start.x = start[0]
            segment_msg.start.y = start[1]
            segment_msg.start.z = start[2]
            segment_msg.end.x = end[0]
            segment_msg.end.y = end[1]
            segment_msg.end.z = end[2]
            markings.segments.append(segment_msg)

        ###################
        # Marking ellipse #
        ###################

        ellipse: sv2dm.MarkingEllipse
        for ellipse in msg.ellipses:
            # Create center point
            center = Point2DStamped(
                header=msg.header,
                point=ellipse.center
            )

            # TODO check math
            diff = ellipse.bb.center.x - ellipse.center.x
            radius = ellipse.bb.size_x // 2 + diff
            side_point = center.copy()
            side_point.x += radius

            # Map point from image onto field plane
            try:
                mapped_center_point = self.ipm.map_point(
                    field,
                    center,
                    output_frame=self._output_frame)
                mapped_side_point = self.ipm.map_point(
                    field,
                    side_point,
                    output_frame=self._output_frame)
                mapped_ellipse = sv3dm.MarkingEllipse()
                mapped_ellipse.center = mapped_center_point.point
                mapped_ellipse.confidence.confidence = ellipse.confidence.confidence
                radius = np.linalg.norm(
                    [
                        mapped_center_point.point.x - mapped_side_point.point.x,
                        mapped_center_point.point.y - mapped_side_point.point.y,
                        mapped_center_point.point.z - mapped_side_point.point.z,
                    ]
                )
                mapped_ellipse.diameter = 2 * radius
                markings.ellipses.append(mapped_ellipse)
            except NoIntersectionError:
                self.get_logger().warn(
                    'Got an ellipse with center point ({},{}) I could not transform.'.format(
                        ellipse.center.x,
                        ellipse.center.x),
                    throttle_duration_sec=5)

        self.obstacles_pub.publish(markings) # TODO

    def callback_field_boundary(self, msg: sv2dm.FieldBoundary):
        field = self.get_field(msg.header.stamp, 0.0)

        field_boundary = sv3dm.FieldBoundary()
        field_boundary.header.stamp = msg.header.stamp
        field_boundary.header.frame_id = self._output_frame
        field_boundary.confidence = msg.confidence

        # Convert points to numpy array
        points_np = np.array([[p.x, p.y] for p in msg.points])

        # Map all points at once from image onto field plane
        points_on_plane = self.ipm.map_points(
            field,
            points_np,
            msg.header,
            output_frame=self._output_frame)

        # Convert numpy array to points
        field_boundary.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points_on_plane]

        self.field_boundary_pub.publish(field_boundary)

    def callback_masks(self, msg: Image, publisher, encoding='8UC1', scale: float = 1.0):
        """
        Map a mask from the input image as a pointcloud on the field plane.

        :param msg: Mask msg type
        :param publisher: Publisher which should be used to publish the mapped point cloud
        :param encoding: Encoding of the input mask. For the exact format see the cv_bride docs.
        :param scale: Downsampling which is applied to the mask before the mapping.
        """
        # Get field plane
        field = self.get_field(msg.header.stamp, 0.0)
        if field is None:
            return

        # Convert subsampled image
        image = cv2.resize(
            self._cv_bridge.imgmsg_to_cv2(msg, encoding),
            (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Get indices for all non 0 pixels (the pixels which should be displayed in the pointcloud)
        point_idx_tuple = np.where(image != 0)

        # Restructure index tuple to a array
        point_idx_array = np.empty((point_idx_tuple[0].shape[0], 3))
        point_idx_array[:, 0] = point_idx_tuple[1] / scale
        point_idx_array[:, 1] = point_idx_tuple[0] / scale

        # Map points
        points_on_plane = self.ipm.map_points(
                    field,
                    point_idx_array,
                    msg.header,
                    output_frame=self._output_frame)

        # Make a pointcloud2 out of them
        pc = create_cloud_xyz32(
            Header(
                stamp=msg.header.stamp,
                frame_id=self._output_frame
            ),
            points_on_plane)

        # Publish point cloud
        publisher.publish(pc)

    def _object_at_bottom_of_image(self, position, thresh):
        """
        Check if the objects y position is at the bottom of the image.

        :param position: Y-position of the object
        :param thresh: Threshold defining the region at the bottom of the image which is
            counted as 'the bottom' as a fraction of the image height
        """
        image_height = self.ipm.get_camera_info().height
        image_height /= max(self.ipm.get_camera_info().binning_y, 1)

        scaled_thresh = thresh * image_height
        return position > scaled_thresh

    def _bb_footpoint(self, bounding_box: BoundingBox2D) -> Point2D:  # TODO rotated bb
        return Point2D(
            x=float(bounding_box.center.position.x),
            y=float(bounding_box.center.position.y + bounding_box.size_y // 2),
        )


def main(args=None):
    rclpy.init(args=args)
    node = SoccerIPM()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    ex.spin()
    node.destroy_node()
    rclpy.shutdown()
