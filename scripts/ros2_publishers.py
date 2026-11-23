import rclpy
from rclpy.node import Node

import numpy as np
import yaml

from tf2_ros import TransformListener, TransformBroadcaster, Buffer
from geometry_msgs.msg import TransformStamped, Point32
from sensor_msgs.msg import PointCloud2, PointField

type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
            (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
            (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

def dtype_to_fields(dtype):
    '''Convert a numpy record datatype into a list of PointFields.
    '''
    datatype=PointField.FLOAT32
    fields = [PointField(name='x', offset=0, datatype=datatype, count=1),
          PointField(name='y', offset=4, datatype=datatype, count=1),
          PointField(name='z', offset=8, datatype=datatype, count=1),
        #   PointField(name='x_norm', offset=12, datatype=datatype, count=1),
        #   PointField(name='y_norm', offset=16, datatype=datatype, count=1),
        #   PointField(name='z_norm', offset=20, datatype=datatype, count=1)
          ]
    
    return fields

def ndarray_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
    '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
    '''
    cloud_arr = cloud_arr.astype(np.float32)
    cloud_msg = PointCloud2()

    itemsize = cloud_arr.itemsize

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = 1
    cloud_msg.width = cloud_arr.shape[0]
    cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = False # assumption
    cloud_msg.point_step = itemsize * cloud_arr.shape[1]
    cloud_msg.row_step = itemsize * cloud_arr.shape[1] * cloud_arr.shape[0]
    cloud_msg.is_dense = True
    cloud_msg.data = cloud_arr.tobytes()

    return cloud_msg

def ndarray_to_tf2(pose = np.array([0,0,0]), quat_obj = np.array([0,0,0,1])):
        t = TransformStamped()

        t.header.stamp = rclpy.time.Time().to_msg()
        # t.header.frame_id = 'tool0'
        # t.child_frame_id = 'camera_depth_optical_frame'

        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = pose[2]

        t.transform.rotation.x = quat_obj[0]
        t.transform.rotation.y = quat_obj[1]
        t.transform.rotation.z = quat_obj[2]
        t.transform.rotation.w = quat_obj[3]

        return t

class TFPublisherNode(Node):
    def __init__(self, node_name='tf_pub'):
        super().__init__(node_name)
        self.tf_talker = TransformBroadcaster(self)

    def write_tf(self, source_frame, target_frame, t):

        t.header.frame_id = source_frame
        t.child_frame_id = target_frame
        
        self.tf_talker.sendTransform(
            t
        )

class PCDPublisherNode(Node):
    def __init__(self, pcd_topic, node_name):
        super().__init__(node_name)
        self.pose_pub = self.create_publisher(
            PointCloud2,
            pcd_topic, 
            10
        )

    def callback(self, data, frame):
        msg = ndarray_to_pointcloud2(data, stamp=rclpy.time.Time().to_msg(), frame_id=frame)
        self.pose_pub.publish(msg)