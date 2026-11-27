import numpy as np
import cv2 as cv
import open3d as o3d
from scipy.spatial.transform import Rotation
import yaml
from typing import Dict, Callable, List

from ros2_subscribers import ImageSubscriber
from ros2_publishers import TFPublisher, PCDPublisher
from sensor_msgs.msg import CameraInfo
import rclpy
from rclpy.node import Node

import threading
import argparse
from os.path import join
from dataclasses import dataclass

# from collections import OrderedDict

@dataclass
class Intrinsics:

    K: np.ndarray = np.identity(3)
    D: np.ndarray = np.zeros(5)
    alpha: int = 0
    xi: int = 0
    w: int = -1
    h: int = -1

    @classmethod
    def load_file(cls, path: str, w: int = 640, h: int = 480):

        path = join(path, "cal_data.yaml")

        with open(path, "r") as f:
            calib = yaml.safe_load(f)

        camera_model = 'pinhole'
        if 'camera_model' in calib:
            camera_model = calib['camera_model']

        K = np.identity(3)
        if isinstance(calib["intrinsics"], list):
            K[0,0] = calib["intrinsics"][0]
            K[1,1] = calib["intrinsics"][1]
            K[0,2] = calib["intrinsics"][2]
            K[1,2] = calib["intrinsics"][3]
        elif isinstance(calib["intrinsics"], dict):
            K[0,0] = calib["intrinsics"]["fx"]
            K[1,1] = calib["intrinsics"]["fy"]
            K[0,2] = calib["intrinsics"]["cx"]
            K[1,2] = calib["intrinsics"]["cy"]
        D = np.zeros(5)

        if camera_model == 'pinhole' and 'distortion' in calib:
            if isinstance(calib["distortion"], list):
                D = np.array(calib["distortion"])
            elif isinstance(calib["distortion"], dict):
                D[0] = calib["distortion"]["k1"]
                D[1] = calib["distortion"]["k2"]
                D[2] = calib["distortion"]["t1"]
                D[3] = calib["distortion"]["t2"]
                D[4] = calib["distortion"]["k3"]

        alpha = 0
        xi = 0
        if camera_model == 'ds' and 'distortion' in calib:
            alpha = calib['distortion']['alpha']
            xi = calib['distortion']['xi']

        return cls(K=K,D=D,alpha=alpha,xi=xi,w=w,h=h)

class CameraParams:

    def __init__(self, intr: Intrinsics | CameraInfo):
        if isinstance(intr, Intrinsics):
            self.K = intr.K
            self.D = intr.D
            self.width, self.height = intr.w, intr.h
            self.alpha, self.xi = intr.alpha, intr.xi
        elif isinstance(intr, CameraInfo):
            self.K = intr.k.reshape(3,3)
            self.D = intr.d
            self.width, self.height = intr.width, intr.height
            self.alpha, self.xi = None, None

class PCDFilter(Node): # name yet to be decided
    def __init__(self, args: argparse.Namespace | Dict[str, str | List[str]]): # can test | more
        super().__init__("pcd_filter_node")

        if args.topic_pcd is None:
            self.get_logger().error(f"No value provided for pcd topic")
            return

        if args.path_handeye is None:
            self.get_logger().error(f"No value provided for handeye calibration result path")
            return

        if not args.path_depth_info is None:
            self.get_logger().info(f"Found path for depth image params. Ignoring depth info topic")
            self.load_cam_params(args.path_depth_info)

        elif args.topic_depth_info is None:
            self.get_logger().error(f"No value provided for depth info topic")
            return

        # maybe can make pcd method in a function, that is called here
        if not args.path_mesh is None:
            self.get_logger().info(f"Found path for mesh file. Ignoring depth topic")
            self.get_logger().info(f"For pointcloud using file {args.path_mesh}")
            self.pcd = o3d.io.read_triangle_mesh(args.path_mesh)
            # read_pcd_file
            self.to_pcd = self.file_to_pcd

        elif args.topic_depth is None:
            self.get_logger().error(f"No value provided for depth topic")
            return

        else:
            self.cam_params = None
            self.depth_sub = ImageSubscriber(
                node=self,
                image_topic=args.topic_depth,
                info_topic=args.topic_depth_info
            )
            self.get_logger().info(f"Using depth topic {args.topic_depth}")
            if args.debug:
                cv.namedWindow("Depth window", cv.WINDOW_NORMAL)
                cv.resizeWindow("Depth window", (640, 480))
            self.load_calib(args.path_handeye)

            if not args.topic_rgb is None:
                self.rgb_sub = ImageSubscriber(
                    node=self,
                    image_topic=args.topic_rgb
                )
                self.get_logger().info(f"Using rgb topic {args.topic_rgb}")
                if args.debug:
                    cv.namedWindow("RGB window", cv.WINDOW_NORMAL)
                    cv.resizeWindow("RGB window", (640, 480))
                self.to_pcd = self.rgbd_to_pcd

            else:
                self.rgb_sub = None
                self.to_pcd = self.depth_to_pcd

        self.tf_pub = TFPublisher(
            node=self
        )

        self.pcd_pub = PCDPublisher(
            node=self,
            pcd_topic = args.topic_pcd
        )

        self.get_logger().info(f"Pointcloud will be published to topic {args.topic_pcd} with frame id {'cam'}")

        self.debug = args.debug

        # freq = 0.5
        # self.timer = self.create_timer(freq, self.callback)

    def callback(self):

        if self.cam_params is None:
            if self.depth_sub.latest_info_msg is None:
                self.get_logger().warning(f"A value for depth camera parameters, has not been recieved from topic: {self.depth_sub.info_sub.topic}")
                return
            self.cam_params = CameraParams(self.depth_sub.latest_info_msg)
            self.get_logger().info(f"Depth camera parameters: {self.cam_params}")

        if self.debug:
            cv.imshow("Depth window", self.depth_sub.latest_frame)

            if not self.rgb_sub is None:
                np_image = self.rgb_sub.latest_frame
                np_image = cv.cvtColor(np_image, cv.COLOR_BGR2RGB)
                cv.imshow("RGB window", np_image)

            key = cv.waitKey(1)
            # if key == 113:
            #     exit

        self.tf_pub.write_tf('world', 'cam', self.T_b_c)

        pcd = self.to_pcd()

        pcd = np.asarray(pcd.points)[np.where(np.asarray(pcd.points)[:,2] < 0.75)]

        self.pcd_pub.callback(pcd, 'cam')
        # or make publishing function

    def depth_to_pcd(self):
        frame = self.depth_sub.latest_frame

        o3d_cam_params = o3d.camera.PinholeCameraIntrinsic(self.cam_params.width, self.cam_params.height, self.cam_params.K)
        o3d_image = o3d.geometry.Image(frame.astype(np.float32))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_image, o3d_cam_params)

        return pcd

    def rgbd_to_pcd(self):
        color_frame = self.rgb_sub.latest_frame
        depth_frame = self.depth_sub.latest_frame

        o3d_cam_params = o3d.camera.PinholeCameraIntrinsic(self.cam_params.width, self.cam_params.height, self.cam_params.K)

        o3d_color_image = o3d.geometry.Image(color_frame.astype(np.float32))
        o3d_depth_image = o3d.geometry.Image(depth_frame.astype(np.float32))
        rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
            color = o3d_color_image,
            depth = o3d_depth_image
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam_params)

        return pcd

    def file_to_pcd(self):
        return self.pcd

    def read_pcd_file(self):
        pass

    def load_cam_params(self, path):
        intrinsics = Intrinsics.load_file(path)
        self.cam_params = CameraParams(self.T_b_w, intrinsics)

    def load_calib(self, path):
        with open(path, 'r') as f:
            calib = yaml.safe_load(f)

        # yaml.add_representer(OrderedDict, represent_dict_order)
        # yaml.dump(d, open('test.yaml', 'w'), default_flow_style=None)

        self.T_b_c = np.identity(4)
        self.T_b_c[:3,:3] = Rotation.from_quat(calib['T_b_c']['xyzw'], scalar_first=False).as_matrix()
        self.T_b_c[:3,3:] = np.array([calib["T_b_c"]["xyz"]]).T

        # self.T_g_c = np.identity(4)
        # T_g_c_quat = np.array([
        #     calib["T_g_c"]["qw"],
        #     calib["T_g_c"]["qx"],
        #     calib["T_g_c"]["qy"],
        #     calib["T_g_c"]["qz"]
        # ])
        # self.T_g_c[:3,:3] = Rotation.from_quat(T_g_c_quat, scalar_first=True).as_matrix()
        # self.T_g_c[:3,3:] = np.array([
        #     calib["T_g_c"]["x"],
        #     calib["T_g_c"]["y"],
        #     calib["T_g_c"]["z"]
        # ]).reshape(3,1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        # prog="pcd_pub",
        # usage="pcd_pub.py [--topic_depth DEPTH_TOPIC] [--topic_depth_info DEPTH_INFO_TOPIC] [--topic_rgb RGB_TOPIC] [--topic_pcd PCD_TOPIC] [--path_depth_info DEPTH_INFO_PATH] [--path_handeye HANDEYE_PATH]",
        # "pcd_pub [--topic_depth DEPTH_TOPIC] [--topic_depth_info DEPTH_INFO_TOPIC] [--topic_rgb RGB_TOPIC] [--topic_pcd PCD_TOPIC] [--path_depth_info DEPTH_INFO_PATH] [--path_handeye HANDEYE_PATH]"
        # "pcd_pub --topic_depth DEPTH_TOPIC --topic_depth_info DEPTH_INFO_TOPIC --topic_rgb RGB_TOPIC --topic_pcd PCD_TOPIC --path_depth_info DEPTH_INFO_PATH --path_handeye HANDEYE_PATH"
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--topic_depth", "--topic_img",
        help="cameras topic to aquire images",
        default="/camera/camera/depth/image_rect_raw",
    )

    parser.add_argument(
        "--topic_pcd", "--pcd",
        help="cameras topic to aquire images",
        default="/points",
    )

    parser.add_argument(
        "--topic_depth_info",
        help="cameras topic to aquire camera info",
        default="/camera/camera/depth/camera_info",
    )

    parser.add_argument(
        "--topic_rgb",
        help="cameras topic to aquire images",
        default="/camera/camera/color/image_raw",
    )

    parser.add_argument(
        "--path_depth_info",
        help="cameras topic to aquire images"
    )

    parser.add_argument(
        "--path_handeye",
        help="cameras topic to aquire images",
        default="params/apriltag_calib.yaml"
    )

    parser.add_argument(
        "--path_mesh",
        help="cameras topic to aquire images",
        # default="datasets/Realdex/Realdex/meshdata/box.obj"
    )

    parser.add_argument(
        '--debug',
        action='store_true'
    )

    args = parser.parse_args()
    rclpy.init()

    pf = PCDFilter(args)
    # mesh = o3d.geometry.PointCloud()

    # depth_node = ImageSubscriberNode(
    #     image_topic=args.topic_depth,
    #     info_topic=args.topic_depth_info,
    #     node_name='img_subscriber'
    # )

    # pcd_node = PCDSubscriberNode(
    #     pcd_topic=args.pcd,
    #     node_name='pcd_subscriber'
    # )

    # pcd_pub_node = PCDPublisherNode(
    #     pcd_topic=args.pcd,
    #     node_name='pcd_publisher'
    # )

    # tf_pub_node = TFPublisherNode(
    #     node_name='obj_points'
    # )

    # executor = rclpy.executors.MultiThreadedExecutor()
    # executor.add_node(img_node)
    # executor.add_node(pcd_pub_node)
    # executor.add_node(tf_pub_node)

    # executor_thread = threading.Thread(target=executor.spin, daemon=True)
    # executor_thread.start()
    
    # img = np.zeros((10,10,3))

    # pcd_model_path = 'data/Realdex/Realdex/meshdata/duck_toy.obj'
    # mesh = o3d.io.read_triangle_mesh(pcd_model_path)
    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()
    # object_points = np.asarray(mesh.vertices)
    # pts_count = int(len(mesh.vertices))
    # object_pcd = mesh.sample_points_uniformly(number_of_points=pts_count, use_triangle_normal=False)
    # object_points = np.asarray(object_pcd.points)
    # object_normals = np.asarray(object_pcd.normals)
    # object_points = np.concat((object_points,object_normals),axis=1)

    # rate = tf_pub_node.create_rate(2)

    # print(f'Subscribing to {args.cam}')
    # print(f'Publishing {args.pcd}')

    exec = rclpy.executors.MultiThreadedExecutor()
    exec.add_node(pf)

    executor_thread = threading.Thread(target=exec.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok():
            pf.callback()
            # exec.spin()
    
    # try:
    #     while rclpy.ok():
    #         t_w_c = np.array([0.4,-0.5,0.34])
    #         # q_w_c = Rotation.from_rotvec(np.array([0,0,0], float) * np.pi/2).as_matrix()
    #         q_w_c = Rotation.from_rotvec(np.array([1,0,0], float) * -np.pi/2).as_matrix()
    #         # q_w_c = q_w_c @ Rotation.from_rotvec(np.array([0,0,1], float) * np.pi/2).as_matrix()
    #         q_w_c = q_w_c @ Rotation.from_rotvec(np.array([0,-0.5,0], float) * np.pi/2).as_matrix()
    #         q_w_c = Rotation.from_matrix(q_w_c).as_quat()
    #         cam_tf = make_tf(t_w_c, q_w_c)

    #         t_c_o = np.array([0, 0, 0.56])
    #         q_c_o = Rotation.from_rotvec(np.array([2,0,0], float) * np.pi/2).as_matrix()
    #         q_c_o = q_c_o @ Rotation.from_rotvec(np.array([0.,-1.3,0], float) * np.pi/2).as_matrix()
    #         q_c_o = q_c_o @ Rotation.from_rotvec(np.array([0,0,1.], float) * np.pi/2).as_matrix()
    #         q_c_o = Rotation.from_matrix(q_c_o).as_quat()
    #         obj_tf = make_tf(t_c_o, q_c_o)
    #         tf_pub_node.write_tf('world', 'cam', cam_tf)
    #         tf_pub_node.write_tf('cam', 'obj_pcd', obj_tf)
    #         pcd_pub_node.callback(object_points, 'obj_pcd')
    #         rate.sleep()
    # try:
    #     while rclpy.ok():
    #         with img_node.lock:
    #             frame = img_node.latest_frame
    #             cam_params = img_node.cam_info
            
    #         tf_pub_node.write_tf('world', 'cam', cam_tf)

    #         if frame is not None and cam_params is not None:
    #             print('Prep values')
    #             img = frame
    #             width = cam_params.width
    #             height = cam_params.height
    #             intrinsic = cam_params.k.reshape(3,3)
    #             print('Making mesh')
    #             o3d_cam_params = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic)
    #             o3d_image = o3d.geometry.Image(frame.astype(np.float32))
    #             mesh = mesh.create_from_depth_image(o3d_image, o3d_cam_params)
    #             mesh = mesh.random_down_sample(0.3)
    #             print('Publishing point cloud')
    #             pcd_pub_node.callback(np.asarray(mesh.points)[np.where(np.asarray(mesh.points)[:,2] < 0.75)], 'cam')
    #             img_node.latest_frame = None
    #             img_node.cam_info = None
            
    #         cv.imshow("Image Window", cv.resize(img, (960, 600)))
            
    #         key = cv.waitKey(1)
    #         if key == 113: # q
    #             break
            
    except KeyboardInterrupt:
        cv.destroyAllWindows()
        pass

    # rclpy.shutdown()
    executor_thread.join()