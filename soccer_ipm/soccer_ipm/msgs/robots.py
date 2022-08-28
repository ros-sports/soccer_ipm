from ipm_interfaces.msg import Point2DStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from rclpy.impl.rcutils_logger import RcutilsLogger
from soccer_ipm.utils import (bb_footpoint, create_field_plane, object_at_bottom_of_image)
import soccer_vision_2d_msgs.msg as sv2dm
import soccer_vision_3d_msgs.msg as sv3dm


def map_robot_array(
        msg: sv2dm.RobotArray,
        ipm: IPM,
        output_frame: str,
        logger: RcutilsLogger,
        footpoint_out_of_image_threshold: float) -> sv3dm.RobotArray:
    """
    Map a given array of 2D goal robot detections onto the field plane.

    :param msg: The 2D message that should be mapped
    :param ipm: An instance of the IPM mapping utility
    :param output_frame: The tf frame of the field
    :param logger: A ros logger to display warnings etc.
    :param footpoint_out_of_image_threshold: Size of the area at the bottom of the image at which
        the object is considered to be only partially visible
    :returns: The robots as 3D cartesian detections in the output_frame
    """
    field = create_field_plane(msg.header.stamp, output_frame)

    robots = sv3dm.RobotArray()
    robots.header.stamp = msg.header.stamp
    robots.header.frame_id = output_frame

    robot: sv2dm.Robot
    for robot in msg.robots:

        # Check if post is not going out of the image at the bottom
        if not object_at_bottom_of_image(
                ipm.get_camera_info(),
                bb_footpoint(robot.bb).y,
                footpoint_out_of_image_threshold):
            # Create footpoint
            footpoint = Point2DStamped(
                header=msg.header,
                point=bb_footpoint(robot.bb)
            )
            # Map point from image onto field plane
            try:
                relative_foot_point = ipm.map_point(
                    field,
                    footpoint,
                    output_frame=output_frame)

                transformed_robot = sv3dm.Robot()
                transformed_robot.attributes = robot.attributes
                transformed_robot.confidence = robot.confidence
                transformed_robot.bb.center.position = relative_foot_point.point
                transformed_robot.bb.size.x = 0.3   # TODO better size estimation
                transformed_robot.bb.size.y = 0.3   # TODO better size estimation
                transformed_robot.bb.size.z = 0.5   # TODO better size estimation
                robots.robots.append(transformed_robot)
            except NoIntersectionError:
                logger.warn(
                    'Got a robot with foot point ({},{}) I could not transform.'.format(
                        footpoint.point.x,
                        footpoint.point.y),
                    throttle_duration_sec=5)
    return robots
