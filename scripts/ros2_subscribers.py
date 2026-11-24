import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.msg import CameraInfo

from tf2_ros import TransformListener, Buffer

# import cv2 as cv
import numpy as np
from cv_bridge import CvBridge

import threading

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
            t = self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time()
            )

            return t

        except Exception as e:
            print(f"TF not available: {e}")
            return None