from ipm_interfaces.msg import Point2DStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from rclpy.impl.rcutils_logger import RcutilsLogger
from soccer_ipm.utils import create_field_plane
import soccer_vision_2d_msgs.msg as sv2dm
import soccer_vision_3d_msgs.msg as sv3dm


def map_ball_array(
        msg: sv2dm.BallArray,
        ipm: IPM,
        output_frame: str,
        logger: RcutilsLogger,
        ball_diameter: float) -> sv3dm.BallArray:
    """
    Map a given array of 2D ball detections onto the field plane.

    :param msg: The 2D message that should be mapped
    :param ipm: An instance of the IPM mapping utility
    :param output_frame: The tf frame of the field
    :param logger: A ros logger to display warnings etc.
    :param ball_diameter: The diameter of the balls that are mapped
    :returns: The balls as 3D cartesian detections in the output_frame
    """
    field = create_field_plane(msg.header.stamp, output_frame, ball_diameter / 2)

    balls_relative = sv3dm.BallArray()
    balls_relative.header.stamp = msg.header.stamp
    balls_relative.header.frame_id = output_frame

    ball: sv2dm.Ball
    for ball in msg.balls:
        ball_point = Point2DStamped(
            header=msg.header,
            point=ball.center)
        try:
            transformed_ball = ipm.map_point(
                field,
                ball_point,
                output_frame=output_frame)

            ball_relative = sv3dm.Ball()
            ball_relative.center = transformed_ball.point
            ball_relative.confidence = ball.confidence
            balls_relative.balls.append(ball_relative)
        except NoIntersectionError:
            logger.warn(
                'Got a ball at ({},{}) I could not transform.'.format(
                    ball.center.x,
                    ball.center.y),
                throttle_duration_sec=5)
    return balls_relative
