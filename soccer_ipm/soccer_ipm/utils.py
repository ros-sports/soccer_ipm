from builtin_interfaces.msg import Time
from ipm_interfaces.msg import PlaneStamped
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import BoundingBox2D, Point2D


def create_field_plane(
        time: Time,
        output_frame: str,
        height_offset: float = 0.0):
    """Create a plane message for a given frame at a given time, with a given height offset."""
    plane = PlaneStamped()
    plane.header.frame_id = output_frame
    plane.header.stamp = time
    plane.plane.coef[2] = 1.0  # Normal in z direction
    plane.plane.coef[3] = -height_offset  # Distance above the ground
    return plane


def object_at_bottom_of_image(
            camera_info: CameraInfo,
            position: float,
            thresh: float) -> bool:
    """
    Check if the objects y position is at the bottom of the image.

    :param camera_info: The current CameraInfo message
    :param position: Y-position of the object
    :param thresh: Threshold defining the region at the bottom of the image which is
        counted as 'the bottom' as a fraction of the image height
    """
    image_height = camera_info.height
    image_height /= max(camera_info.binning_y, 1)

    scaled_thresh = thresh * image_height
    return position > scaled_thresh


def bb_footpoint(bounding_box: BoundingBox2D) -> Point2D:
    """
    Return the footpoint of a given bounding box.

    :param bounding_box: The bounding box
    :returns: The footpoint (aka. the bottom center)
    """
    return Point2D(
        x=float(bounding_box.center.position.x),
        y=float(bounding_box.center.position.y + bounding_box.size_y // 2),
    )


def compose(f, g):
    """Composes two functions into a new one."""
    return lambda *args, **kwargs: f(g(*args, **kwargs))
