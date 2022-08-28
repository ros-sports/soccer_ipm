from ipm_interfaces.msg import Point2DStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from rclpy.impl.rcutils_logger import RcutilsLogger
from soccer_ipm.utils import (bb_footpoint, create_field_plane, object_at_bottom_of_image)
import soccer_vision_2d_msgs.msg as sv2dm
import soccer_vision_3d_msgs.msg as sv3dm


def map_goalpost_array(
        msg: sv2dm.GoalpostArray,
        ipm: IPM,
        output_frame: str,
        logger: RcutilsLogger,
        footpoint_out_of_image_threshold: float) -> sv3dm.GoalpostArray:
    """
    Map a given array of 2D goal post detections onto the field plane.

    :param msg: The 2D message that should be mapped
    :param ipm: An instance of the IPM mapping utility
    :param output_frame: The tf frame of the field
    :param logger: A ros logger to display warnings etc.
    :param footpoint_out_of_image_threshold: Size of the area at the bottom of the image at which
        the object is considered to be only partially visible
    :returns: The posts as 3D cartesian detections in the output_frame
    """
    field = create_field_plane(msg.header.stamp, output_frame)

    # Create new message
    goalposts_relative_msg = sv3dm.GoalpostArray()
    goalposts_relative_msg.header.stamp = msg.header.stamp
    goalposts_relative_msg.header.frame_id = output_frame

    # Transform goal posts
    goal_post_in_image: sv2dm.Goalpost
    for goal_post_in_image in msg.posts:
        # Check if post is not going out of the image at the bottom
        ipm.get_camera_info()
        if not object_at_bottom_of_image(
                ipm.get_camera_info(),
                bb_footpoint(goal_post_in_image.bb).y,
                footpoint_out_of_image_threshold):
            # Create footpoint
            footpoint = Point2DStamped(
                header=msg.header,
                point=bb_footpoint(goal_post_in_image.bb)
            )
            # Map point from image onto field plane
            try:
                relative_foot_point = ipm.map_point(
                    field,
                    footpoint,
                    output_frame=output_frame)

                post_relative = sv3dm.Goalpost()
                post_relative.attributes = goal_post_in_image.attributes
                post_relative.bb.center.position = relative_foot_point.point
                post_relative.bb.size.x = 0.1  # TODO better size estimation
                post_relative.bb.size.y = 0.1  # TODO better size estimation
                post_relative.bb.size.z = 1.5  # TODO better size estimation
                post_relative.confidence = goal_post_in_image.confidence
                goalposts_relative_msg.posts.append(post_relative)
            except NoIntersectionError:
                logger.warn(
                    'Got a post with foot point ({},{}) I could not transform.'.format(
                        footpoint.point.x,
                        footpoint.point.y),
                    throttle_duration_sec=5)
    return goalposts_relative_msg
