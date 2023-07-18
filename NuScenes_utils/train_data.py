from nuscenes.nuscenes import NuScenes
import nuscenes.nuscenes as nu
from nuscenes.map_expansion.arcline_path_utils import discretize_lane

from NuScenes_utils.map import Map
from NuScenes_utils.preprocessing import DatasetBuilder
from NuScenes_utils.scene import AgentCentricScene

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import dill
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
import random
import argparse
import os


class DatasetCreator:
    def __init__(self, args):
        self.args = args
        self.parent_dir = Path(__file__).parents[1].resolve()

    def get_train_val(self):
        train_loader = DatasetSaver(args=self.args, data_type=torch.FloatTensor)
        save_dir = f'{self.parent_dir}/dataset_for_train/train_set'
        self.check_dir(save_dir)
        train_iter = self.loader(train_loader)
        idx = 0
        for batch_data in tqdm(train_iter, desc='Creating Train dataset'):
            for i in range(len(batch_data)):
                data = batch_data[i]
                torch.save(data, f'{save_dir}/data_{idx}.pth')
                idx += 1

        val_loader = DatasetSaver(args=self.args, validation=True, data_type=torch.FloatTensor)
        save_dir = f'{self.parent_dir}/dataset_for_train/val_set'
        self.check_dir(save_dir)
        val_iter = self.loader(val_loader)
        idx = 0
        for batch_data in tqdm(val_iter, desc='Creating Validation Dataset'):
            for i in range(len(batch_data)):
                data = batch_data[i]
                torch.save(data, f'{save_dir}/data_{idx}.pth')
                idx += 1

    def get_test(self):
        test_loader = DatasetSaver(args=self.args, isTrain=False, data_type=torch.FloatTensor)

        save_dir = f'{self.parent_dir}/dataset_for_train/test_set'
        self.check_dir(save_dir)
        test_iter = self.loader(test_loader, batch_size=1)
        idx = 0
        for batch_data in tqdm(test_iter, desc='Creating Train dataset'):
            for i in range(len(batch_data)):
                data = batch_data[i]
                torch.save(data, f'{save_dir}/data_{idx}.pth')
                idx += 1

    def loader(self, loader, batch_size=100):
        iterator = DataLoader(loader, batch_size=batch_size, shuffle=False,
                              num_workers=self.args.num_cores, collate_fn=self.seq_collate_typeA)
        return iterator

    def seq_collate_typeA(self, data):
        return data

    def check_dir(self, dirs):
        if dirs != '' and not os.path.exists(dirs):
            os.makedirs(dirs)


class DatasetSaver(Dataset):
    def __init__(self, args, data_type=torch.FloatTensor, isTrain=True, validation=False):
        super().__init__()
        self.args = args
        self.isTrain = isTrain

        exp_type = 'train' if isTrain else 'test'
        self.validation = validation

        self.nusc = NuScenes(version='v1.0-trainval', dataroot='dataset_raw', verbose=True)

        self.map = Map(args, self.nusc)

        self.dtype = data_type
        self.target_sample_period = self.args.target_sample_period  # 2Hz
        self.obs_len = int(self.args.past_horizon_seconds * self.args.target_sample_period)  # 2*2=4 points
        self.pred_len = int(self.args.future_horizon_seconds * self.args.target_sample_period)  # 6*2 = 12 points
        self.min_obs_len = int(self.args.min_past_horizon_seconds * self.args.target_sample_period)  # 1.5*2 = 3 points

        self.path_resol = self.args.path_resol  # 1
        self.path_len_f = self.args.max_path_len_forward  # 80
        self.path_len_b = self.args.max_path_len_backward  # 10
        self.num_pos_f = int(self.args.max_path_len_forward / self.path_resol)  # 80/1=80
        self.num_pos_b = int(self.args.max_path_len_backward / self.path_resol)  # 10/1=10
        self.num_max_paths = self.args.num_max_paths  # 10

        self.limit_range = self.args.limit_range  # 30
        self.is_random_path_order = self.args.is_random_path_order

        self.parent_dir = Path(__file__).parents[1].resolve()
        save_path = f'{self.parent_dir}/dataset_preprocessed/' \
                    f'{self.args.past_horizon_seconds}sec_{self.args.future_horizon_seconds}sec'
        self.check_dir(save_path)
        file_name = f'nuscenes_{exp_type}_cat{self.args.category_filtering_method}.cpkl'

        # Check preprocessed file
        if not os.path.exists(f'{save_path}/{file_name}'):
            builder = DatasetBuilder(self.args, map=self.map, isTrain=self.isTrain)
            builder.make_preprocessed_data(f'{save_path}/{file_name}')

        with open(f'{save_path}/{file_name}', 'rb') as f:
            dataset = dill.load(f, encoding='latin1')

        if self.isTrain:
            if self.validation:
                data = dataset[1]
            else:
                data = dataset[0]

            self.data = []  # update, 211015 (b/o exp365)
            for _, scene in enumerate(tqdm(data, desc='refactoring data')):
                self.data += self.refactoring(scene)  # Creating target from every object in scene

        else:
            self.data = dataset[2]
        self.num_scene = len(self.data)
        pass

    def refactoring(self, scene):

        """
        Convert Ego-centric (or AV-centric) data to Agent-centric data
        """

        target = []

        # current sample (current time)
        num_total_agents = scene.num_agents  # Considered object in scene
        sample = self.nusc.get('sample', scene.sample_token)
        lidar_sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ref_ego_pose = self.nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
        R_e2g = Quaternion(ref_ego_pose['rotation']).rotation_matrix  # Convert Ego to Global
        R_g2e = np.linalg.inv(R_e2g)  # Inverse
        translation_g_e = np.array(ref_ego_pose['translation']).reshape(1, 3)

        # all agent's trajectories
        agent_ids = np.zeros(shape=(1, num_total_agents))
        trajectories = np.full(shape=(self.obs_len + self.pred_len, num_total_agents, 3),
                               fill_value=np.nan)  # 4, 12, x, 3
        bounding_boxes = np.full(shape=(num_total_agents, 8, 3), fill_value=np.nan)

        # get trajectories (Position of each object from prev to next) & bounding boxes
        for idx, track_id in enumerate(scene.agent_dict):
            agent_ids[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]
            bounding_boxes[idx, :, :] = scene.agent_dict[track_id].bbox_3d().T

        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len - 1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]  # ID for each object in scene
        bounding_boxes = bounding_boxes[valid_flag, :, :]

        # find agents who have valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        bounding_boxes = bounding_boxes[valid_flag, :, :]

        # transform to global coordinate system
        trajectories_g = np.copy(trajectories)
        for agent_id in range(agent_ids.shape[1]):
            trajectories_g[:, agent_id, :] = self.map.transform_pc_inv(R_e2g, translation_g_e,
                                                                       trajectories[:, agent_id, :])
            # Transform local object coordinate

        # for all agents
        for agent_id in range(agent_ids.shape[1]):  # Create target from each object in scene
            agent_track_id = scene.id_2_token_lookup[agent_ids[0, agent_id]]
            agent = scene.agent_dict[agent_track_id]

            if agent_track_id == 'EGO':
                R_g2a = R_g2e
                R_a2g = R_e2g
            else:
                ann = self.nusc.get('sample_annotation', agent_track_id)
                R_a2g = Quaternion(ann['rotation']).rotation_matrix
                R_g2a = np.linalg.inv(R_a2g)

            trans_a = trajectories_g[self.obs_len - 1, agent_id, :].reshape(1, 3)  # Global trajectory over observation

            # skip if the target agent doesn't have full future trajectory
            FLAG = np.min(trajectories_g[self.obs_len:, agent_id, 0]) > -1000  # -1000 means no next
            if not FLAG:
                continue

            # bounding_boxes
            bboxes_g = np.copy(bounding_boxes)
            bboxes_g[agent_id, :, :] = self.map.transform_pc_inv(R_a2g, trans_a, bounding_boxes[agent_id, :, :])
            # Convert bounding boxes coordinate into global

            # trajectories
            trajectories_a = np.copy(trajectories_g)
            for aa in range(agent_ids.shape[1]):
                trajectories_a[:, aa, :] = self.map.transform_pc(R_g2a, trans_a, trajectories_g[:, aa, :])

            agent_sample = AgentCentricScene(sample_token=scene.sample_token, agent_token=agent_track_id,
                                             city_name=scene.city_name)
            agent_sample.target_agent_index = agent_ids[0, agent_id]
            agent_sample.trajectories = trajectories_a
            agent_sample.bboxes = bboxes_g
            agent_sample.R_a2g = R_a2g
            agent_sample.R_g2a = R_g2a
            agent_sample.trans_g = trans_a
            agent_sample.agent_ids = agent_ids
            agent_sample.possible_lanes = agent.possible_lanes
            target.append(agent_sample)

        return target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.isTrain:
            """
                    obs_traj_ta : seq_len x 1 x dim
                    future_traj_ta : seq_len x 1 x dim
                    obs_traj_ngh : seq_len x num_neighbors x dim
                    future_traj_ngh : seq_len x num_neighbors x dim
                    map : 1 x 3 x map_size x map_size
                    num_neighbors : 1
                    valid_neighbor :  1 (bool)
                    possible_paths : seq_len x num_max_paths x dim
            """

            # current scene data

            obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, maps, num_neighbors, valid_neighbor, possible_paths, \
                lane_label = self.extract_data_from_scene(self.data[idx])

            obs_traj_ta = torch.from_numpy(obs_traj_ta).type(self.dtype)
            future_traj_ta = torch.from_numpy(future_traj_ta).type(self.dtype)
            obs_traj_ngh = torch.from_numpy(obs_traj_ngh).type(self.dtype)
            future_traj_ngh = torch.from_numpy(future_traj_ngh).type(self.dtype)
            maps = torch.from_numpy(maps).permute(2, 0, 1).type(self.dtype)
            maps = torch.unsqueeze(maps, dim=0)
            possible_paths = torch.from_numpy(possible_paths).type(self.dtype)
            lane_label = torch.from_numpy(lane_label).type(self.dtype)

            return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
                maps, torch.tensor(num_neighbors), torch.tensor(valid_neighbor), possible_paths, lane_label
        else:
            obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor, \
                possible_lane, lane_label, agent_ids, agent_samples, scene = self.get_test_data(idx)
            return [obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor,
                    possible_lane, lane_label, agent_ids, agent_samples, scene]

    def extract_data_from_scene(self, scene, isTrain=True):

        """
        Extract training data from Scene
        """

        agent_ids = scene.agent_ids
        target_agent_index = scene.target_agent_index
        trajectories = scene.trajectories
        bboxes = scene.bboxes
        possible_lanes = scene.possible_lanes

        R_g2a = scene.R_g2a
        R_a2g = scene.R_a2g
        trans_g = scene.trans_g  # Global Transpose matrix

        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len - 1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]

        # split into target agent and neighbors
        num_agents = agent_ids.shape[1]
        trajectory_ta = np.full(shape=(self.obs_len + self.pred_len, 1, 3), fill_value=np.nan)
        trajectories_ngh = []
        idx = np.argwhere(agent_ids[0, :] == target_agent_index)[0][0]
        for l in range(num_agents):
            if l == idx:
                trajectory_ta[:, :, :] = np.copy(trajectories[:, l, :].reshape(self.obs_len + self.pred_len, 1, 3))
            else:
                trajectories_ngh.append(np.copy(trajectories[:, l, :].reshape(self.obs_len + self.pred_len, 1, 3)))

        num_neighbors = len(trajectories_ngh)
        if num_neighbors == 0:
            trajectories_ngh = np.full(shape=(self.obs_len + self.pred_len, 1, 3), fill_value=np.nan)
            valid_neighbor = False
            num_neighbors += 1
        else:
            trajectories_ngh = np.concatenate(trajectories_ngh, axis=1)
            valid_neighbor = True

        # calc speed and heading
        trajectory_ta_ext = self.calc_speed_heading(trajectory_ta)  # Available on Paper
        trajectories_ngh_ext = []
        for n in range(trajectories_ngh.shape[1]):
            trajectories_ngh_ext.append(
                self.calc_speed_heading(trajectories_ngh[:, n, :].reshape(self.obs_len + self.pred_len, 1, 3)))
        trajectories_ngh_ext = np.concatenate(trajectories_ngh_ext, axis=1)

        # split into observation and future
        obs_traj_ta = np.copy(trajectory_ta_ext[:self.obs_len])
        future_traj_ta = np.copy(trajectory_ta_ext[self.obs_len:])
        obs_traj_ngh = np.copy(trajectories_ngh_ext[:self.obs_len])
        future_traj_ngh = np.copy(trajectories_ngh_ext[self.obs_len:])

        # remove 'nan' in observation
        obs_traj_ta = self.remove_nan(obs_traj_ta)
        obs_traj_ngh = self.remove_nan(obs_traj_ngh)

        # NOTE : currently not used
        maps = np.zeros(shape=(10, 10, 3))

        # candidate lanes
        possible_paths, lane_label = self.get_lane_coords(possible_lanes, R_g2a, trans_g, scene.city_name)

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, maps, num_neighbors, valid_neighbor, \
            possible_paths, lane_label

    def calc_speed_heading(self, trajectory):

        """
            trajectory : seq_len x batch x 3
        """

        # params
        seq_len, batch, dim = trajectory.shape

        # speed (m/s) and heading (rad)
        traj = np.copy(np.squeeze(trajectory)[:, :2])
        pos_diff = np.zeros_like(traj)
        pos_diff[1:, :] = traj[1:] - traj[:-1]

        # speed
        speed_mps = self.target_sample_period * np.sqrt(np.sum(pos_diff ** 2, axis=1)).reshape(seq_len, 1, 1)

        # heading
        heading_rad = np.arctan2(pos_diff[:, 1], pos_diff[:, 0]).reshape(seq_len, 1, 1)

        return np.concatenate([speed_mps, heading_rad, trajectory], axis=2)

    def remove_nan(self, seq):

        """
            seq : seq_len x batch x 2
        """
        seq_copy = np.copy(seq)
        for i in range(seq.shape[1]):
            cur_seq = np.copy(seq[:, i, :])
            if np.count_nonzero(np.isnan(cur_seq[:-self.min_obs_len])) > 0:
                seq_copy[:-self.min_obs_len, i, :] = 0.0
        return seq_copy

    def convert_to_egocentric(self, _obs_traj, _future_traj, _pred_traj, agent_ids, agent_samples):

        """
        Convert Agent-centric data to Ego-centric (or AV-centric) data for visualization
        """

        best_k = _pred_traj.shape[0]
        num_agents = len(agent_samples)

        # extend dims
        z_axis = np.expand_dims(_future_traj[:, :, 2].reshape(self.pred_len, num_agents, 1), axis=0)
        z_axis = np.repeat(z_axis, best_k, axis=0)
        _pred_traj = np.concatenate([_pred_traj, z_axis], axis=3)

        # ego-vehicle R & T
        idx = np.argwhere(agent_ids == 0)[0][0]
        assert (idx == 0)
        R_g2e = agent_samples[idx].R_g2a
        trans_g_e = agent_samples[idx].trans_g

        obs_traj, future_traj, pred_traj_k = [], [], []
        for i in range(num_agents):

            # ego-vehicle
            if agent_ids[i] == 0:
                obs = _obs_traj[:, i, :].reshape(self.obs_len, 1, 3)
                future = _future_traj[:, i, :].reshape(self.pred_len, 1, 3)
                preds = _pred_traj[:, :, i, :].reshape(best_k, self.pred_len, 1, 3)

                obs_traj.append(obs)
                future_traj.append(future)
                pred_traj_k.append(preds)

            else:
                R_a2g = agent_samples[i].R_a2g
                trans_g_a = agent_samples[i].trans_g

                obs = _obs_traj[:, i, :]
                future = _future_traj[:, i, :]
                preds = _pred_traj[:, :, i, :]

                obs = self.map.transform_pc(R_g2e, trans_g_e, self.map.transform_pc_inv(R_a2g, trans_g_a, obs))
                future = self.map.transform_pc(R_g2e, trans_g_e, self.map.transform_pc_inv(R_a2g, trans_g_a, future))

                preds_k = []
                for k in range(best_k):
                    pred = self.map.transform_pc(R_g2e, trans_g_e,
                                                 self.map.transform_pc_inv(R_a2g, trans_g_a, preds[k, :, :]))
                    preds_k.append(np.expand_dims(pred, axis=0))
                preds = np.concatenate(preds_k, axis=0)

                obs_traj.append(np.expand_dims(obs, axis=1))
                future_traj.append(np.expand_dims(future, axis=1))
                pred_traj_k.append(np.expand_dims(preds, axis=2))

        obs_traj = np.concatenate(obs_traj, axis=1)
        future_traj = np.concatenate(future_traj, axis=1)
        pred_traj_k = np.concatenate(pred_traj_k, axis=2)

        return obs_traj, future_traj, pred_traj_k

    def get_lane_coords(self, possible_lanes, R_g2a, trans_g, location):

        """
        Get equally-spaced positions of a centerline from lane token sequence
        """

        filter = np.array([0.5, 0, 0.5])
        target_spacing = np.arange(0, self.path_len_f, self.path_resol)
        min_path_len = 5.0  # meter

        # get lane coordinates
        possible_paths = []
        for _, tok_seq in enumerate(possible_lanes):
            path = []

            # discretize and global2ego transform
            for __, tok in enumerate(tok_seq):
                lane_record = self.map.nusc_maps[location].get_arcline_path(tok)
                coords = np.array(discretize_lane(lane_record, resolution_meters=0.05))
                path.append(coords)
            path = np.concatenate(path, axis=0)
            path_agent_centric = self.map.transform_pc(R_g2a, trans_g, path)[:, :2]

            # find target segment
            start_idx = np.argmin(np.sum(np.abs(path_agent_centric[:, :2]), axis=1))
            path_agent_centric = path_agent_centric[start_idx:]
            path_len = path_agent_centric.shape[0]
            if path_len < int(min_path_len / self.path_resol):
                continue

            # sample equally-spaced
            point_dist = np.zeros(shape=(path_agent_centric.shape[0]))
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1]) ** 2, axis=1))
            sorted_index = np.searchsorted(np.cumsum(point_dist), target_spacing, side='right')
            chk = sorted_index < path_len
            sorted_index = sorted_index[chk]
            path_agent_centric = path_agent_centric[sorted_index]

            # centerline quality
            seq_len = path_agent_centric.shape[0]
            point_dist = self.path_resol * np.ones(shape=seq_len)
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1]) ** 2, axis=1))

            # smoothing filter
            if np.max(point_dist) > 1.1 * self.path_resol or np.min(point_dist) < 0.9 * self.path_resol:
                path_agent_centric_x_avg = np.convolve(path_agent_centric[:, 0], filter, mode='same').reshape(seq_len,
                                                                                                              1)
                path_agent_centric_y_avg = np.convolve(path_agent_centric[:, 1], filter, mode='same').reshape(seq_len,
                                                                                                              1)
                path_agent_centric_avg = np.concatenate([path_agent_centric_x_avg, path_agent_centric_y_avg], axis=1)

                chk = point_dist > 1.1 * self.path_resol
                path_agent_centric[chk] = path_agent_centric_avg[chk]

                chk = point_dist < 0.9 * self.path_resol
                path_agent_centric[chk] = path_agent_centric_avg[chk]

            # length of current lane
            path_len = path_agent_centric.shape[0]
            if path_len < int(min_path_len / self.path_resol):
                path_agent_centric = np.full(shape=(self.num_pos_f, 2), fill_value=np.nan)
                path_len = path_agent_centric.shape[0]

            # increase length of current lane
            if path_len < self.num_pos_f:
                num_repeat = self.num_pos_f - path_len
                delta = (path_agent_centric[1:] - path_agent_centric[:-1])[-1].reshape(1, 2)
                delta = np.repeat(delta, num_repeat, axis=0)
                delta[0, :] += path_agent_centric[-1]
                padd = np.cumsum(delta, axis=0)
                path_agent_centric = np.concatenate([path_agent_centric, padd], axis=0)

            possible_paths.append(np.expand_dims(path_agent_centric, axis=1))
            assert (path_agent_centric.shape[0] == self.num_pos_f)

        # add fake lanes
        num_repeat = 0
        if self.num_max_paths > len(possible_paths):
            num_repeat = self.num_max_paths - len(possible_paths)
            for i in range(num_repeat):
                possible_paths.append(np.full(shape=(self.num_pos_f, 1, 2), fill_value=np.nan))

        # NOTE : 'is_random_path_order' should be 1
        # randomize the order of lanes
        indices = [idx for idx in range(self.num_max_paths)]
        if self.is_random_path_order == 1:
            random.shuffle(indices)

        possible_paths_random = []
        for _, idx in enumerate(indices):
            possible_paths_random.append(possible_paths[idx])
        possible_paths_random = np.concatenate(possible_paths_random, axis=1)

        # reference lane index
        label = np.zeros(shape=(1, self.num_max_paths))
        if num_repeat == self.num_max_paths:
            best_match_idx = indices[0]
        else:
            best_match_idx = np.argwhere(np.array(indices) == 0)[0][0]
        label[0, best_match_idx] = 1

        return possible_paths_random, label

    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nu.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)

    def check_dir(self, dirs):
        if dirs != '' and not os.path.exists(dirs):
            os.makedirs(dirs)

    def get_test_data(self, idx):
        scene = self.data[idx]
        agent_samples = self.refactoring(scene)
        obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, agent_ids, possible_paths, \
            lane_label = [], [], [], [], [], [], [], [], [], []

        for i in range(len(agent_samples)):
            # agent_index
            agent_id = agent_samples[i].target_agent_index
            # current sample
            _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _num_neighbors, _valid_neighbor, _possible_paths, \
                _lane_label = self.extract_data_from_scene(agent_samples[i])

            _obs_traj = torch.from_numpy(_obs_traj).type(self.dtype)
            _future_traj = torch.from_numpy(_future_traj).type(self.dtype)
            _obs_traj_ngh = torch.from_numpy(_obs_traj_ngh).type(self.dtype)
            _future_traj_ngh = torch.from_numpy(_future_traj_ngh).type(self.dtype)
            _map = torch.from_numpy(_map).permute(2, 0, 1).type(self.dtype)
            _map = torch.unsqueeze(_map, dim=0)
            _possible_paths = torch.from_numpy(_possible_paths).type(self.dtype)
            _lane_label = torch.from_numpy(_lane_label).type(self.dtype)

            obs_traj.append(_obs_traj)
            future_traj.append(_future_traj)
            obs_traj_ngh.append(_obs_traj_ngh)
            future_traj_ngh.append(_future_traj_ngh)
            map.append(_map)
            num_neighbors.append(_num_neighbors)
            valid_neighbor.append(_valid_neighbor)
            agent_ids.append(agent_id)
            possible_paths.append(_possible_paths)
            lane_label.append(_lane_label)

        _len = [objs for objs in num_neighbors]
        cum_start_idx = [0] + np.cumsum(_len).tolist()
        seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        obs_traj = torch.cat(obs_traj, dim=1)
        future_traj = torch.cat(future_traj, dim=1)
        obs_traj_ngh = torch.cat(obs_traj_ngh, dim=1)
        future_traj_ngh = torch.cat(future_traj_ngh, dim=1)
        map = torch.cat(map, dim=0)
        seq_start_end = torch.LongTensor(seq_start_end)
        valid_neighbor = np.array(valid_neighbor)
        agent_ids = np.array(agent_ids)
        possible_paths = torch.cat(possible_paths, dim=1)
        lane_label = torch.cat(lane_label, dim=0)

        return obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor, \
            possible_paths, lane_label, agent_ids, agent_samples, scene
