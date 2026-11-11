import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
from scipy.spatial.transform import Rotation

from utils.utils import HandParams

class GraspAnyting(Dataset):

    _joint_names = ['WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 
                    'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 
                    'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 
                    'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 
                    'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
    # _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
    #                                    -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
    #                                    -0.5237, 0.])
    # _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
    #                                    1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
    #                                    0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
    #                                    0.5237, 1.])

    # _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425])
    # _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427])


    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, dataset, object_nors) -> None:
        super(GraspAnyting, self).__init__()

        self.hand_params = HandParams()

        self.len = len(dataset['metadata'])

        self.hand_poses = [data['translations'] for data in dataset['metadata']]
        self.joint_values = [data['joint_positions'] for data in dataset['metadata']]
        self.hand_rotations = [data['rotations'] for data in dataset['metadata']]
        self.item_names = [data['object_name'] for data in dataset['metadata']]

        # self.hand_poses = torch.tensor(self.hand_poses)
        # self.joint_values = torch.tensor(self.joint_values)
        # self.hand_rotations = torch.tensor(self.hand_rotations)

        # grasp_config_filename = 'data/dex_hand_limits.yaml'
        # with open(grasp_config_filename, encoding="utf-8") as grasp_config_file:
        #     joint_limit_yaml = yaml.load(grasp_config_file, Loader=yaml.FullLoader)

        # self.joint_name_order_ = joint_limit_yaml['joint_names']
        # joint_min_values = dict(zip(joint_limit_yaml['joint_names'], joint_limit_yaml['urdf_joint_values']['joint_min_value']))
        # joint_max_values = dict(zip(joint_limit_yaml['joint_names'], joint_limit_yaml['urdf_joint_values']['joint_max_value']))

        # self._joint_angle_lower = torch.tensor([joint_min_values[f'rh_{joint}'] for joint in self._joint_names])
        # self._joint_angle_upper = torch.tensor([joint_max_values[f'rh_{joint}'] for joint in self._joint_names])

        # self._global_trans_lower = torch.tensor(joint_limit_yaml['global_lower'])
        # self._global_trans_upper = torch.tensor(joint_limit_yaml['global_upper'])

        self.object_nors = object_nors

    # def set_joint_limits(self, lower_limit, upper_limit):
    #     self._joint_angle_lower = lower_limit.to(self.device)
    #     self._joint_angle_upper = upper_limit.to(self.device)

    # def set_global_limits(self, lower_limit, upper_limit):
    #     self._global_trans_lower = lower_limit.to(self.device)
    #     self._global_trans_upper = upper_limit.to(self.device)

    # def trans_normalize(self, global_trans: torch.Tensor):
    #     global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
    #     global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
    #     return global_trans_norm

    # def trans_denormalize(self, global_trans: torch.Tensor):
    #     global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
    #     global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
    #     global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
    #     return global_trans_denorm

    # def angle_normalize(self, joint_angle: torch.Tensor):
    #     joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
    #     joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
    #     return joint_angle_norm

    # def angle_denormalize(self, joint_angle: torch.Tensor):
    #     joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
    #     joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
    #     joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
    #     return joint_angle_denorm

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):

        hand_point = self.hand_poses[index]
        joint_values = self.joint_values[index]
        hand_rotation = self.hand_rotations[index].T
        
        non_range = self.hand_params._joint_angle_lower > joint_values
        if non_range.any():
            joint_values[non_range] = self.hand_params._joint_angle_lower[non_range]
        non_range = self.hand_params._joint_angle_upper < joint_values
        if non_range.any():
            joint_values[non_range] = self.hand_params._joint_angle_upper[non_range]
        

        normalized_hand_point = self.hand_params.trans_normalize(hand_point)
        normalized_joint_values = self.hand_params.angle_normalize(joint_values)
        hand_point_and_joints = torch.cat(
            [normalized_hand_point, normalized_joint_values], 
            dim=0).requires_grad_(True) # why combine both?? figure out if why or can saprate?

        item = self.item_names[index]
        geo_v_and_v_norm = torch.tensor(self.object_nors[item])

        rand_rot = Rotation.random().as_matrix()

        object_points_trans = np.einsum('mn, kn->km', hand_rotation, geo_v_and_v_norm[:,:3])
        vertex_normal = np.einsum('mn, kn->km', hand_rotation, geo_v_and_v_norm[:,3:])

        # object_points_trans = np.einsum('mn, kn->km', rand_rot, geo_v_and_v_norm[:,:3])
        # vertex_normal = np.einsum('mn, kn->km', rand_rot, geo_v_and_v_norm[:,3:])

        cam_tran = None

        rand = np.random.permutation(len(object_points_trans))
        vertex_normal = vertex_normal[rand[:2048]]
        object_points_trans = object_points_trans[rand[:2048]]

        # hand_point_and_joints = hand_point_and_joints.to(torch.float32)
        # hand_rotation = torch.tensor(hand_rotation, dtype=torch.float64)
        # nor = torch.tensor(nor, dtype=torch.float64)
        # object_points_trans = torch.tensor(object_points_trans, dtype=torch.float64)

        data = {
            'x': hand_point_and_joints, # hand pose and joints
            'pos': object_points_trans, # objects ?? if nor is object mesh points this could be those points rotated???
            'scene_rot_mat': hand_rotation, # hand rotation
            # 'cam_tran': cam_tran, # camera pose, but empty?
            'scene_id': item, # object name
            'normal': vertex_normal, # ?? this could be an objects mesh points
            # 'offset': 1
        }

        feat = np.concatenate([vertex_normal], axis=-1)
        data['feat'] = torch.tensor(feat)

        for key in data:
            if not torch.is_tensor(data[key]) and type(data[key]) is not str:
                data[key] = torch.tensor(data[key])

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
