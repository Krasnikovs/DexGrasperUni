import torch
import numpy as np
import pickle
import os
from typing import Dict
from scipy.spatial.transform import Rotation
import open3d as o3d
import yaml

from models.ddpm import DDPM
from models.unet import UNetModel
from utils.utils import HandParams
from optimizer.grasp_with_object import GraspWithObject

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformListener, Buffer
from tf2_ros import TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import argparse
import threading

def tf2_to_numpy(tf):
    
    trans = np.array([
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z
    ])

    rot = np.array([
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w
    ])

    return trans, rot

def numpy_to_tf2(pose = np.array([0,0,0]), quat_obj = np.array([0,0,0,1])):
        tf = TransformStamped()

        tf.header.stamp = rclpy.time.Time().to_msg()

        pose = pose.flatten().astype(float)

        tf.transform.translation.x = pose[0]
        tf.transform.translation.y = pose[1]
        tf.transform.translation.z = pose[2]

        quat_obj = quat_obj.flatten().astype(float)

        tf.transform.rotation.x = quat_obj[0]
        tf.transform.rotation.y = quat_obj[1]
        tf.transform.rotation.z = quat_obj[2]
        tf.transform.rotation.w = quat_obj[3]

        return tf

class TFPublisherNode(Node):
    def __init__(self, node_name='tf_pub'):
        super().__init__(node_name)
        self.tf_talker = TransformBroadcaster(self)

    def write_tf(self, source_frame, target_frame, tf):

        tf.header.frame_id = source_frame
        tf.child_frame_id = target_frame
        
        self.tf_talker.sendTransform(
            tf
        )

class JointStatePublisherNode(Node):
    def __init__(self, pose_topic, node_name):
        super().__init__(node_name)
        self.pose_pub = self.create_publisher(
            JointState,
            pose_topic, 
            10
        )

    def callback(self, names, data):
        msg = JointState()
        msg.name = names
        msg._position = data.flatten()
        self.pose_pub.publish(msg)
    
class PCDSubscriberNode(Node):
    def __init__(self, pose_topic, node_name):
        super().__init__(node_name)
        self.latest_msg = None
        self.create_subscription(
            PointCloud2,
            pose_topic, 
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        self.latest_msg = point_cloud2.read_points_numpy(msg)

class TFSubscriberNode(Node):
    def __init__(self, node_name='tf_reader'):
        super().__init__(node_name)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def read_tf(self, source_frame, target_frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=source_frame,
                source_frame=target_frame,
                time=rclpy.time.Time()
            )

            return tf2_to_numpy(tf)

        except Exception as e:
            print(f"TF not available: {e}")
            return None

def collate_fn_squeeze_pcd_batch(batch_data) -> Dict:
    """ General collate function used for dataloader.
    This collate function is used for point-transformer
    """
    offset, count = [], 0
    for item in batch_data['pos']:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_data['offset'] = offset

    batch_data['pos'] = torch.func.rearrange(batch_data['pos'], 'b n c -> (b n) c')
    batch_data['feat'] = torch.func.rearrange(batch_data['feat'], 'b n c -> (b n) c')
    
    return batch_data

def saveWeights(model, URL = './test0.pth', saveStateOnly = True):
    print("Saving weights...")
    if saveStateOnly:
        torch.save(model.state_dict(), URL)
    else:
        torch.save(model, URL)
    print("Done saving")

def loadWeights(model, URL = './test0.pth', loadOnlyWeights = True):
    print(f"Loading weights from {URL}...")
    
    if loadOnlyWeights:
        state_dict = torch.load(URL, weights_only=loadOnlyWeights)
        model.load_state_dict(state_dict)
    else:
        model = torch.load(URL, weights_only=loadOnlyWeights)
    print('Model loaded')
    return model

def load_data(path = 'data/Realdex/Realdex'):
    print('Loading data')
    scene_path = os.path.join(path, 'object_pcds_nors.pkl')
    data_path = os.path.join(path, 'realdex_shadowhand_downsample.pt')

    object_nors = pickle.load(open(scene_path, 'rb'))
    grasp_data = torch.load(data_path)

    print('Done loading')
    return grasp_data, object_nors

def grasp_pub(model, pcd, pcd_position, device=None, item_name: str = 'cylinder'):
    print('Started grasp')

    model.eval()
    hl = HandParams()
    # _joint_angle_lower, _joint_angle_upper = hl.get_joint_limits()
    # _global_trans_lower, _global_trans_upper = hl.get_global_limits()
    # model.set_joint_limits(_joint_angle_lower, _joint_angle_upper)
    # model.set_global_limits(_global_trans_lower, _global_trans_upper)
    # model.optimizer.set_joint_limits(_joint_angle_lower, _joint_angle_upper)
    # model.optimizer.set_global_limits(_global_trans_lower, _global_trans_upper)

    hand_point = torch.tensor(np.zeros(3))
    joint_values = torch.tensor(np.zeros(24))
    
    normalized_hand_point = hl.trans_denormalize(global_trans=hand_point.cpu()).cuda()
    normalized_joint_values = hl.angle_denormalize(joint_angle=joint_values.cpu()).cuda()
    
    hand_point_joints = torch.cat(
                [normalized_hand_point, normalized_joint_values], 
                dim=0).requires_grad_(True).unsqueeze(0)
    
    # hand_rotation = np.expand_dims(np.identity(3),0)
    hand_rotation = np.expand_dims(Rotation.from_quat(np.array([0, 0, 0, 1]), scalar_first=False).as_matrix(),0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(pcd)
    # mesh = mesh.random_down_sample(0.5)
    T_w_o = np.identity(4)
    T_w_o[:3,3:] = pcd_position[0].reshape(3,1)
    object_rotation = Rotation.from_quat(pcd_position[1]).as_matrix()
    T_w_o[:3,:3] = object_rotation
    mesh.transform(T_w_o)
    mesh.estimate_normals(fast_normal_computation = False)

    object_points = np.expand_dims(np.asarray(mesh.points), axis=0)
    object_normals = np.expand_dims(np.asarray(mesh.normals), axis=0)

    model.eps_model.set_scene_model(c = 3, num_points = object_points.shape[1])

    data = {
        'x': hand_point_joints.to(torch.float32),
        'pos': torch.tensor(object_points).to(torch.float32),
        'normal': torch.tensor(object_normals).to(torch.float32),
        'feat': torch.tensor(object_normals).to(torch.float32),
        'scene_rot_mat': torch.tensor(hand_rotation).to(torch.float32),
        'scene_id': item_name
    }

    data = collate_fn_squeeze_pcd_batch(data)
    for key in data:
        if torch.is_tensor(data[key]):
            data[key] = data[key].to(device)

    outputs = model.sample(data).squeeze(1)[:, -1, :].to(torch.float64)

    outputs[:, 3:] = hl.angle_denormalize(joint_angle=outputs[:, 3:].cpu()).to(device)
    outputs[:, :3] = hl.trans_denormalize(global_trans=outputs[:, :3].cpu()).to(device)

    model.optimizer.hand_model.get_keypoints(outputs.to(torch.float32))
    predicted_links = model.optimizer.hand_model.current_status
    predicted_link_tips = [[link, predicted_links[link]._matrix[:,:3,3:]] for link in predicted_links if link.find('tip') != -1]

    pred_hand_point = outputs[:, :3].cpu().detach().numpy().flatten()
    pred_hand_joints = outputs[:, 3:].cpu().detach().numpy().flatten()

    hand_points = model.optimizer.hand_model.get_surface_points(outputs.to(torch.float32))

    hand_pcd = o3d.geometry.PointCloud()

    hand_pcd.points = o3d.utility.Vector3dVector(hand_points.detach().cpu().numpy().squeeze())

    o3d.visualization.draw_geometries([hand_pcd, mesh])

    print(f'Item {item_name}')
    print(f'Item translate {pcd_position[0]}')
    print(f'Item rotation {pcd_position[1]}')
    print(f'Translate {pred_hand_point}')
    print(f'Joints {pred_hand_joints}')
    for link, value in predicted_link_tips:
        print(f'{link}: {value.cpu().detach().numpy().flatten()}')

    return item_name, outputs, hl._joint_names, predicted_link_tips

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="get_data",
        usage="get_data.py [--joint_state JOINT_STATE] [--pcd PCD] [--tf]",
    )

    parser.add_argument(
        "--joint_state",
        default="/desired_joint_states",
    )

    parser.add_argument(
        "--pcd",
        help="cameras topic to aquire images",
        default="/points",
        # default="/camera/camera/color/image_raw",
        # default="/camera/camera/depth/color/points",
    )

    parser.add_argument(
        "--paramsPath",
        default="data/all.yaml",
    )

    parser.add_argument(
        "--model",
        default="data/models/model_epoch_699-26_10_13.pth",
    )
    parser.add_argument(
        "--item",
        default="cylinder",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Using device: {torch.cuda.get_device_name(device)}')
        print(f'Using device: {device}')
    else:
        device = torch.device("cpu")
        print('Using cpu')

    # grasp_data, object_nors = load_data('/home/gustavskrasnikovs/repos/off_shute/data/Realdex/Realdex')

    # grasp_data['metadata'] = [data for data in grasp_data['metadata'] if data['object_name'] == 'cylinder']

    # data = GraspAnyting(grasp_data, object_nors)

    # dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    params = yaml.safe_load(open(args.paramsPath, 'r'))

    model = UNetModel(params['UNetModel'], device).to(device)
    model = DDPM(model, False, device, params['DDPM']).to(device)
    optim = GraspWithObject(params['grasp_with_object'], device)
    model.set_optimizer(optim)

    load_model = True
    if load_model:
        model = loadWeights(model, URL=args.model, loadOnlyWeights=True)

    rclpy.init()

    js_node_pub = JointStatePublisherNode(
        pose_topic=args.joint_state,
        node_name='model_states'
    )

    pcd_node_sub = PCDSubscriberNode(
        pose_topic = args.pcd,
        node_name = 'pcd_for_model'
    )

    tf_node_sub = TFSubscriberNode(
        node_name = 'tf_sub'
    )

    tf_node_pub = TFPublisherNode(
        node_name = 'tf_pub'
    )


    # object, pose_and_joints, joint_name_order = grasp_pub(model, device=device, item_name='cylinder')
    rate = tf_node_sub.create_rate(2)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(js_node_pub)
    executor.add_node(pcd_node_sub)
    executor.add_node(tf_node_sub)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    object_tf = None
    object = None
    item = args.item
    source_tf = 'world'
    target_tf = 'obj_pcd'

    print(f'Subscribing to {args.pcd}')
    print(f'Publishing {args.joint_state}')
    try:
        while rclpy.ok():
            pcd_msg = pcd_node_sub.latest_msg
            object_tf = tf_node_sub.read_tf(source_tf, target_tf)

            if object_tf is None:
                rate.sleep()
            
            if pcd_msg is not None and object_tf is not None:
                # print('Generating grasp')
                if pcd_msg.data.shape == 0:
                    continue

                if object is None:
                    object, pose_and_joints, joint_name_order, pred_links = grasp_pub(model, device=device, pcd=pcd_msg, pcd_position=object_tf, item_name=item)
                    joint_name_order = [f'rh_{joint}' for joint in joint_name_order]

                for link in pred_links:
                    tf = numpy_to_tf2(link[1].cpu().detach().numpy())
                    tf_node_pub.write_tf('world', link[0], tf)
                js_node_pub.callback(joint_name_order, pose_and_joints[:, 3:].cpu().detach().numpy())
                pcd_node_sub.latest_msg = None
                object_tf = None
    except KeyboardInterrupt:
        # rclpy.shutdown()
        pass

    # rclpy.shutdown()
    print('done')