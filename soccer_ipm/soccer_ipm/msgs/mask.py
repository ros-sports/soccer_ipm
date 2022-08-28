import cv2
from cv_bridge import CvBridge
from ipm_library.ipm import IPM
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from soccer_ipm.utils import create_field_plane
from std_msgs.msg import Header


cv_bridge = CvBridge()


def map_masks(
        msg: Image,
        ipm: IPM,
        output_frame: str,
        encoding='8UC1',
        scale: float = 1.0) -> PointCloud2:
    """
    Map a mask from the input image as a pointcloud on the field plane.

    :param msg: Mask msg type
    :param encoding: Encoding of the input mask. For the exact format see the cv_bride docs.
    :param scale: Downsampling which is applied to the mask before the mapping.
    :returns: The projected point cloud
    """
    # Get field plane
    field = create_field_plane(msg.header.stamp, output_frame)
    if field is None:
        return

    # Convert subsampled image
    image = cv2.resize(
        cv_bridge.imgmsg_to_cv2(msg, encoding),
        (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # Get indices for all non 0 pixels (the pixels which should be displayed in the pointcloud)
    point_idx_tuple = np.where(image != 0)

    # Restructure index tuple to a array
    point_idx_array = np.empty((point_idx_tuple[0].shape[0], 3))
    point_idx_array[:, 0] = point_idx_tuple[1] / scale
    point_idx_array[:, 1] = point_idx_tuple[0] / scale

    # Map points
    points_on_plane = ipm.map_points(
                field,
                point_idx_array,
                msg.header,
                output_frame=output_frame)

    # Make a pointcloud2 out of them
    pc = create_cloud_xyz32(
        Header(
            stamp=msg.header.stamp,
            frame_id=output_frame
        ),
        points_on_plane)
    return pc
