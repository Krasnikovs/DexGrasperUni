import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.msg import CameraInfo

from tf2_ros import TransformListener, Buffer

import cv2 as cv
import numpy as np
# from cv_bridge import CvBridge

import threading

cv_type_mappings = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),
	
    # for bayer image (based on cv_bridge.cpp)
	"bayer_rggb8":	(np.uint8,  1),
	"bayer_bggr8":	(np.uint8,  1),
	"bayer_gbrg8":	(np.uint8,  1),
	"bayer_grbg8":	(np.uint8,  1),
	"bayer_rggb16":	(np.uint16, 1),
	"bayer_bggr16":	(np.uint16, 1),
	"bayer_gbrg16":	(np.uint16, 1),
	"bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
}

def imgmsg_to_ndarray(msg):
	if not msg.encoding in cv_type_mappings:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
	
	dtype_class, channels = cv_type_mappings[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.fromstring(msg.data.tobytes(), dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

	if channels == 1:
		data = data[...,0]
	return data

class ImageSubscriber:
    def __init__(self, node: Node, image_topic: str, info_topic: str = None):

        # self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_msg = None

        self.latest_info_msg = None

        self.img_sub = node.create_subscription(
            Image,
            image_topic,
            self.listener_callback,
            10)
        
        self.info_sub = None
        if not info_topic is None:
            self.info_sub = node.create_subscription(
                CameraInfo,
                info_topic,
                self.info_listener_callback,
                10)

    def listener_callback(self, msg):

        # frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame = imgmsg_to_ndarray(msg)
        with self.lock:
            self.latest_frame = frame
            self.latest_msg = msg

    def info_listener_callback(self, msg):
        self.latest_info_msg = msg

class PCDSubscriber:
    def __init__(self, node: Node, pcd_topic: str):

        # self.lock = threading.Lock()
        self.latest_cloud = None

        self.subscription = node.create_subscription(
            PointCloud2,
            pcd_topic,
            self.listener_callback,
            10)

    def listener_callback(self, msg):

        # with self.lock:
        self.latest_cloud = msg

class TFSubscriber:
    def __init__(self, node: Node):

        # node.tf_buffer = Buffer()
        self.tf_listener = TransformListener()

    def read_tf(self, source_frame, target_frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time()
            )

            return tf

        except Exception as e:
            print(f"TF not available: {e}")
            return None

class ImageSubscriberNode(Node):
    def __init__(self, image_topic, info_topic, node_name):
        super().__init__(node_name)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_msg = None

        self.cam_info = None

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.listener_callback,
            10)
        
        self.info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self.info_listener_callback,
            10)

    def listener_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        with self.lock:
            self.latest_frame = frame
            self.latest_msg = msg

    def info_listener_callback(self, msg):
        self.cam_info = msg

class PCDSubscriberNode(Node):
    def __init__(self, pcd_topic, node_name):
        super().__init__(node_name)

        self.lock = threading.Lock()
        self.latest_frame = None

        self.subscription = self.create_subscription(
            PointCloud2,
            pcd_topic,
            self.listener_callback,
            10)

    def listener_callback(self, msg):

        with self.lock:
            self.latest_frame = msg

class TFSubscriberNode(Node):
    def __init__(self, node_name='tf_reader'):
        super().__init__(node_name)

        # self.source_frame = source_frame
        # self.target_frame = target_frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

    def read_tf(self, source_frame, target_frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time()
            )

            return tf

        except Exception as e:
            print(f"TF not available: {e}")
            return None