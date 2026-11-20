import numpy as np
import cv2 as cv
import open3d as o3d
from scipy.spatial.transform import Rotation

from ros2_subscribers import *
from ros2_publishers import *

import rclpy
import threading
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="get_data",
        usage="get_data.py [--source src-frame] [--target target-frame] [--output_path <path to colleted data>] [--cam CAM] [--cam_info CAM_INFO] [--pcd PCD]",
    )

    parser.add_argument(
        "--cam",
        help="cameras topic to aquire images",
        # default="/basler_dart/camera_0/image_raw",
        # default="/camera/camera/color/image_raw",
        default="/camera/camera/depth/image_rect_raw",
    )

    parser.add_argument(
        "--pcd",
        help="cameras topic to aquire images",
        default="/points",
        # default="/camera/camera/color/image_raw",
        # default="/camera/camera/depth/color/points",
    )

    parser.add_argument(
        "--cam_info",
        help="cameras topic to aquire camera info",
        # default="none",
        # default="/camera/camera/color/camera_info",
        default="/camera/camera/depth/camera_info",
    )

    args = parser.parse_args()
    rclpy.init()
    mesh = o3d.geometry.PointCloud()

    img_node = ImageSubscriberNode(
        image_topic=args.cam,
        info_topic=args.cam_info,
        node_name='img_subscriber'
    )

    # pcd_node = PCDSubscriberNode(
    #     pcd_topic=args.pcd,
    #     node_name='pcd_subscriber'
    # )

    pcd_pub_node = PCDPublisherNode(
        pcd_topic=args.pcd,
        node_name='pcd_publisher'
    )

    tf_pub_node = TFPublisherNode(
        node_name='obj_points'
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(img_node)
    executor.add_node(pcd_pub_node)
    executor.add_node(tf_pub_node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    img = np.zeros((10,10,3))

    

    
    # q_c_o = Rotation.random().as_matrix()
    # q_c_o = Rotation.from_matrix(q_c_o).as_quat()
    
    pcd_model_path = 'data/Realdex/Realdex/meshdata/duck_toy.obj'
    mesh = o3d.io.read_triangle_mesh(pcd_model_path)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    # object_points = np.asarray(mesh.vertices)
    pts_count = int(len(mesh.vertices))
    object_pcd = mesh.sample_points_uniformly(number_of_points=pts_count, use_triangle_normal=False)
    object_points = np.asarray(object_pcd.points)
    object_normals = np.asarray(object_pcd.normals)
    object_points = np.concat((object_points,object_normals),axis=1)
    # object_points = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices), use_triangle_normal=True)
    # object_points = np.asarray(object_points.points)
    # object_points = np.loadtxt('/home/gustavskrasnikovs/repos/off_shute/data/monster_0_points.csv', delimiter=',')
    # object_points = np.loadtxt('data/cylinder_points.csv', delimiter=',')


    rate = tf_pub_node.create_rate(2)

    print(f'Subscribing to {args.cam}')
    print(f'Publishing {args.pcd}')
    try:
        while rclpy.ok():
            t_w_c = np.array([0.4,-0.5,0.34])
            # q_w_c = Rotation.from_rotvec(np.array([0,0,0], float) * np.pi/2).as_matrix()
            q_w_c = Rotation.from_rotvec(np.array([1,0,0], float) * -np.pi/2).as_matrix()
            # q_w_c = q_w_c @ Rotation.from_rotvec(np.array([0,0,1], float) * np.pi/2).as_matrix()
            q_w_c = q_w_c @ Rotation.from_rotvec(np.array([0,-0.5,0], float) * np.pi/2).as_matrix()
            q_w_c = Rotation.from_matrix(q_w_c).as_quat()
            cam_tf = make_tf(t_w_c, q_w_c)

            t_c_o = np.array([0, 0, 0.56])
            q_c_o = Rotation.from_rotvec(np.array([2,0,0], float) * np.pi/2).as_matrix()
            q_c_o = q_c_o @ Rotation.from_rotvec(np.array([0.,-1.3,0], float) * np.pi/2).as_matrix()
            q_c_o = q_c_o @ Rotation.from_rotvec(np.array([0,0,1.], float) * np.pi/2).as_matrix()
            q_c_o = Rotation.from_matrix(q_c_o).as_quat()
            obj_tf = make_tf(t_c_o, q_c_o)
            tf_pub_node.write_tf('world', 'cam', cam_tf)
            tf_pub_node.write_tf('cam', 'obj_pcd', obj_tf)
            pcd_pub_node.callback(object_points, 'obj_pcd')
            rate.sleep()
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
        # cv.destroyAllWindows()
        pass

    # rclpy.shutdown()
    executor_thread.join()