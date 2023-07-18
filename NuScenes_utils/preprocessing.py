from NuScenes_utils.agent import Agent
from NuScenes_utils.scene import Scene

import nuscenes.nuscenes as nu
import nuscenes.utils.splits
import nuscenes.utils.data_classes as dc

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from tqdm import tqdm
import dill
import sys
import copy


class DatasetBuilder:
    def __init__(self, args, map, isTrain=True):
        self.nusc = map.nusc
        self.map = map
        self.args = args
        self.isTrain = isTrain

        self.val_ratio = args.val_ratio

        self.test_sample_tokens = []
        self.val_sample_tokens = []
        self.train_sample_tokens = []
        self.split_train_val_test()

        sample_tokens = self.train_sample_tokens if self.isTrain else self.test_sample_tokens
        sample_tokens = sample_tokens[:100]

        #  Parallel Preprocessing
        preprocessing = Preprocessing(args, map, sample_tokens, isTrain)
        self.data_iter = DataLoader(dataset=preprocessing, batch_size=100, shuffle=False,
                                    num_workers=self.args.num_cores, collate_fn=self.seq_collate)

    def make_preprocessed_data(self, file_path):
        # set seed
        np.random.seed(1)
        scene_list_train, scene_list_val, scene_list_test = [], [], []
        for scene in tqdm(self.data_iter, desc='Preprocessing Progress'):

            for i in range(len(scene)):
                if isinstance(scene[i], np.ndarray):
                    continue
                if self.isTrain:
                    if np.random.rand(1) < self.val_ratio:
                        scene_list_val.append(scene[i])
                    else:
                        scene_list_train.append(scene[i])
                else:
                    scene_list_test.append(scene[i])

        with open(file_path, 'wb') as f:
            dill.dump([scene_list_train, scene_list_val, scene_list_test], f, protocol=dill.HIGHEST_PROTOCOL)
        print('>> {%s} is created .. ' % file_path)

    def scene_to_sample_tokens(self, scene):
        samples = nu.traverse_linked_list(self.nusc, self.nusc.get('sample', scene['first_sample_token']),
                                          'sample', 'next', inclusive=True)
        return [sample['token'] for sample in samples]

    def split_train_val_test(self):

        """
        Split scenes into train/valid/test according to Trajectron++ (ECCV, 2020)
        """

        scene_blacklist = [499, 515, 517]

        # split scene names
        train_scenes = nuscenes.utils.splits.train
        # val_scenes = []
        test_scenes = nuscenes.utils.splits.val

        # assign ID to each scene
        train_scene_inds = [self.nusc.scene_name_to_scene_idx[_] for _ in train_scenes]
        # val_scene_inds = []
        test_scene_inds = [self.nusc.scene_name_to_scene_idx[_] for _ in test_scenes]

        # remove scenes in black_list
        for _, ids in enumerate(scene_blacklist):
            if ids in train_scene_inds:
                train_scene_inds.remove(ids)

            # if (id in val_scene_inds):
            #     val_scene_inds.remove(id)

            if ids in test_scene_inds:
                test_scene_inds.remove(ids)

        # all the sample tokens related to train/val/test scenes

        for ti in train_scene_inds:
            self.train_sample_tokens.extend(self.scene_to_sample_tokens(self.nusc.scene[ti]))

        for tei in test_scene_inds:
            self.test_sample_tokens.extend(self.scene_to_sample_tokens(self.nusc.scene[tei]))

    def seq_collate(self, scene):
        return scene


class Preprocessing(Dataset):

    def __init__(self, args, map, token, isTrain):
        super().__init__()
        # basic settings
        exp_type = 'train' if isTrain else 'test'

        # load nuscenes loader
        self.nusc = map.nusc
        self.map = map
        self.model_name = args.model_name  # HLS
        self.token = token

        self.nuscenes_config(args)  # Generate idx for train & testing

        self.dataset_path = os.path.join(args.dataset_path, exp_type)
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)  # 4
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)  # 12
        self.min_num_agents = args.min_num_agents  # 2
        self.target_sample_period = args.target_sample_period  # 2
        self.scene_accept_prob = args.scene_accept_prob  # .33

        self.val_ratio = args.val_ratio
        self.min_past_horizon_seconds = args.min_past_horizon_seconds
        self.min_future_horizon_seconds = args.min_future_horizon_seconds
        self.category_filtering_method = args.category_filtering_method

    def __getitem__(self, i):
        tok = self.token[i]

        def transform_box(box, ego_pose):
            """
            transform to ego pose
            """
            box.transform_to_pose(ego_pose)

        # current sample (current time)
        sample = self.nusc.get('sample', tok)
        sample_token = sample['token']  # get token, same with tok
        location = self.nusc.get('log', self.nusc.get('scene', sample['scene_token'])['log_token'])['location']

        # corresponding lidar_sample_data
        lidar_sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        # sequential lidar_sample_data
        # Retrieve next data
        lidar_future_sample_data = self.traverse_linked_list(lidar_sample_data, 'sample_data', 'next')

        # Retrieve prev data
        lidar_past_sample_data = self.traverse_linked_list(lidar_sample_data, 'sample_data', 'prev')
        lidar_all_sample_data = lidar_past_sample_data + [lidar_sample_data] + lidar_future_sample_data

        # Retrieve lidar token
        lidar_tokens = [_['token'] for _ in lidar_all_sample_data]

        # sequential Lidar timestamps relative to now.
        timestamp_now = lidar_sample_data['timestamp']
        lidar_timestamps = [_['timestamp'] for _ in lidar_all_sample_data]
        lidar_timestamps_relative = [(_ - timestamp_now) / 1e6 for _ in
                                     lidar_timestamps]  # Time stamp relative to current

        # not enough observation length
        if lidar_timestamps_relative[0] > self.target_past_times[0]:
            # Length of lidar past recording Vs. Past observ
            return np.asarray(['Not Qualified'])

        # not enough future length
        if lidar_timestamps_relative[-1] < self.target_future_times[-1]:
            # length of lidar future recording Vs. Future observ
            return np.asarray(['Not Qualified'])

        # # ------------------------------------
        # Ego vehicle trajectory
        # #-------------------------------------

        # World extrinsic + intrinsics of rear axle on driving car.
        ego_pose_now = self.get_ego_pose(lidar_sample_data['ego_pose_token'])
        ego_poses_all = [self.get_ego_pose(_['ego_pose_token']) for _ in lidar_all_sample_data]
        ego_boxes_all = [dc.EgoBox.from_pose(_) for _ in ego_poses_all]

        # Interpolate boxes at the target times (global coordinates).
        ego_boxes_past_interp, lidar_tokens_past = self.interpolate_boxes_to_times(ego_boxes_all,
                                                                                   lidar_timestamps_relative,
                                                                                   lidar_tokens,
                                                                                   self.target_past_times)
        ego_boxes_future_interp, lidar_tokens_future = self.interpolate_boxes_to_times(ego_boxes_all,
                                                                                       lidar_timestamps_relative,
                                                                                       lidar_tokens,
                                                                                       self.target_future_times)
        ego_boxes_interp = ego_boxes_past_interp + ego_boxes_future_interp

        lidar_now_token = lidar_tokens_past[-1]
        lidar_token_seq = lidar_tokens_past + lidar_tokens_future

        # Create agent dictionary for the current scene
        agent_dict = {}
        agent_dict['EGO'] = Agent(types='VEHICLE', track_id='EGO', obs_len=self.obs_len, pred_len=self.pred_len)

        for idx in range(len(ego_boxes_interp)):
            agent_dict['EGO'].trajectory_global_coord[idx, 0] = self.target_all_times[idx]
            agent_dict['EGO'].trajectory_global_coord[idx, 1] = ego_boxes_interp[idx].center[0]
            agent_dict['EGO'].trajectory_global_coord[idx, 2] = ego_boxes_interp[idx].center[1]
            agent_dict['EGO'].trajectory_global_coord[idx, 3] = ego_boxes_interp[idx].center[2]

        # From global coord. system to SDV coord. system at now, Third Step
        _ = [transform_box(_, ego_pose_now) for _ in ego_boxes_interp]

        for idx in range(len(ego_boxes_interp)):
            agent_dict['EGO'].trajectory[idx, 0] = self.target_all_times[idx]
            agent_dict['EGO'].trajectory[idx, 1] = ego_boxes_interp[idx].center[0]
            agent_dict['EGO'].trajectory[idx, 2] = ego_boxes_interp[idx].center[1]
            agent_dict['EGO'].trajectory[idx, 3] = ego_boxes_interp[idx].center[2]

        agent_dict['EGO'].wlh = ego_boxes_past_interp[-1].wlh
        agent_dict['EGO'].yaw = ego_boxes_past_interp[-1].get_yaw()
        agent_dict['EGO'].yaw_prev = ego_boxes_past_interp[-2].get_yaw()

        # # ------------------------------------
        # Neighboring Agents
        # #-------------------------------------
        num_agents = 1
        ann_tokens = sample['anns']
        for ann_token in ann_tokens:  # Create agent for each moving car
            ann = self.nusc.get('sample_annotation', ann_token)

            category = ann['category_name']
            if len(ann['attribute_tokens']):
                attribute = self.nusc.get('attribute', ann['attribute_tokens'][0])['name']
            else:
                continue

            if not self.check_valid_agent(category, attribute):  # Check whether moving car
                continue

            # ------------------------
            # consider agents who have valid trajectory length
            # ------------------------

            # Keyframed annotations.
            anns_future = self.traverse_linked_list(ann, 'sample_annotation', 'next')  # Retreive next annotation
            anns_past = self.traverse_linked_list(ann, 'sample_annotation', 'prev')  # Retreive prev annotation
            full_annotations = anns_past + [ann] + anns_future

            # Timestamps
            full_ann_timestamps = [self.nusc.get('sample', _['sample_token'])['timestamp'] for _ in full_annotations]
            full_ann_timestamps_relative = [(_ - timestamp_now) / 1e6 for _ in full_ann_timestamps]

            # checking trajectory length
            if full_ann_timestamps_relative[0] > -1 * self.min_past_horizon_seconds:
                # Not enough agent past annotations. Skipping
                continue

            if full_ann_timestamps_relative[-1] < self.min_future_horizon_seconds:
                # Not enough agent future annotations. Skipping
                continue

            # define target time indices
            for i in range(self.target_past_times.size):
                check = np.array(full_ann_timestamps_relative) < self.target_past_times[i]
                if np.count_nonzero(check):
                    target_past_times = self.target_past_times[i:]
                    break

            for i in range(self.target_future_times.size - 1, -1, -1):
                check = np.array(full_ann_timestamps_relative) > self.target_future_times[i]
                if np.count_nonzero(check):
                    target_future_times = self.target_future_times[:i + 1]
                    break

            # Included in an agent dictionary
            agent = Agent(types=category, attribute=attribute, track_id=ann['token'], obs_len=self.obs_len,
                          pred_len=self.pred_len)

            # ------------------------
            # extract trajectory, bbox, and yaw info
            # ------------------------
            full_ann_boxes = [self.nusc.get_box(_['token']) for _ in
                              full_annotations]  # Get interpolated anns at non-keyframe

            # Interpolate boxes at the target times (global coordinates.)
            # Matching annotations to target time
            ann_boxes_past_interp, _ = self.interpolate_boxes_to_times(full_ann_boxes, full_ann_timestamps_relative,
                                                                       lidar_tokens, target_past_times)
            #  Originally return: interpolated box & lidar token, but only take interpolated box
            ann_boxes_future_interp, _ = self.interpolate_boxes_to_times(full_ann_boxes, full_ann_timestamps_relative,
                                                                         lidar_tokens, target_future_times)
            full_ann_boxes_interp = ann_boxes_past_interp + ann_boxes_future_interp
            full_ann_target_all_times = np.concatenate([target_past_times, target_future_times])

            sort_indices = np.searchsorted(self.target_all_times, full_ann_target_all_times)
            for _ in range(sort_indices.size):
                idx = sort_indices[_]
                agent.trajectory_global_coord[idx, 0] = full_ann_target_all_times[_]
                agent.trajectory_global_coord[idx, 1] = full_ann_boxes_interp[_].center[0]
                agent.trajectory_global_coord[idx, 2] = full_ann_boxes_interp[_].center[1]
                agent.trajectory_global_coord[idx, 3] = full_ann_boxes_interp[_].center[2]
                #  agent.trajectory_global_coord: time, x, y, z

            # Annotation boxes in ego_pose now or lidar_pose now coordinates.
            _ = [transform_box(_, ego_pose_now) for _ in full_ann_boxes_interp]

            for _ in range(sort_indices.size):
                idx = sort_indices[_]
                agent.trajectory[idx, 0] = full_ann_target_all_times[_]
                agent.trajectory[idx, 1] = full_ann_boxes_interp[_].center[0]
                agent.trajectory[idx, 2] = full_ann_boxes_interp[_].center[1]
                agent.trajectory[idx, 3] = full_ann_boxes_interp[_].center[2]

            agent.wlh = ann_boxes_past_interp[-1].wlh
            agent.yaw = ann_boxes_past_interp[-1].get_yaw()
            agent.yaw_prev = ann_boxes_past_interp[-2].get_yaw()

            # ------------------------
            # store in an agent dictionary
            # ------------------------
            agent_dict[ann['token']] = agent
            num_agents += 1

        if num_agents < self.min_num_agents:
            return np.asarray(['Not Qualified'])

        # update agents information
        for idx, key in enumerate(agent_dict):
            agent_dict[key].agent_id = idx
            agent_dict[key].heading_from_traj()  # direction : Eq (25)
            agent_dict[key].calc_speed(self.target_sample_period)  # Eq (24)
            agent_dict[key].trajectory_curvature()  # update curvature (curve)
            agent_dict[key].update_status()  # either start stop, turn
            if 'HLS' in self.model_name:
                agent_dict[key].possible_lanes = self.map.find_possible_lanes(agent_dict[key], lidar_now_token,
                                                                              location)
                if len(agent_dict[key].possible_lanes) > 0:
                    agent_dict[key].possible_lanes = self.map.find_best_lane(agent_dict[key].possible_lanes,
                                                                             agent_dict[key].trajectory_global_coord[:,
                                                                             1:],
                                                                             location,
                                                                             self.obs_len)

        # save in the scene  -> Scene as RECORDER
        scene = Scene(sample_token=sample_token, lidar_token_seq=lidar_token_seq, agent_dict=agent_dict,
                      city_name=location)
        scene.make_id_2_token_lookup()
        return scene

    def nuscenes_config(self, args):

        """
        Samples are at 2Hz
        LIDAR is at 20Hz

        target vehicle category and attribute are based on official prediction benchmark
        (https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/prediction_tutorial.ipynb)
        """

        self.target_vehicle_category = ['vehicle.car', 'vehicle.bus.rigid', 'vehicle.truck', 'vehicle.bus.bendy',
                                        'vehicle.emergency.police', 'vehicle.construction']
        self.target_vehicle_attribute = ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked']

        # Predict future_horizon_seconds in the future with past_horizon_seconds.
        past_horizon_seconds = args.past_horizon_seconds  # 2
        future_horizon_seconds = args.future_horizon_seconds  # 6

        # Configuration for temporal interpolation.
        target_sample_period_past = args.target_sample_period  # 2
        target_sample_period_future = args.target_sample_period  # 2
        target_sample_frequency_past = 1. / target_sample_period_past  # 1/2 sec
        target_sample_frequency_future = 1. / target_sample_period_future  # 1/2

        self.target_past_times = -1 * np.arange(0, past_horizon_seconds, target_sample_frequency_past)[::-1]  # -6, 0
        self.target_past_times[np.isclose(self.target_past_times, 0.0, atol=1e-5)] = 1e-8
        self.target_future_times = np.arange(target_sample_frequency_future, future_horizon_seconds
                                             + target_sample_frequency_future, target_sample_frequency_future)

        self.target_all_times = np.concatenate([self.target_past_times, self.target_future_times])
        pass

    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nu.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)

    def interpolate_boxes_to_times(self, boxes, box_timestamps_relative, lidar_tokens, relative_times_interp):
        return nu.interpolate_boxes_to_times(boxes, box_timestamps_relative, lidar_tokens,
                                             relative_times_interp)

    def get_ego_pose(self, pose_token):
        return self.nusc.get('ego_pose', pose_token)

    def check_valid_agent(self, category, attribute):

        if self.category_filtering_method == 0:  # Selected filter = 0
            # ------------------------
            # According to Trajectron++ (ECCV, 2020)
            # ------------------------
            if 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                return True
            else:
                return False

        elif self.category_filtering_method == 1:
            # ------------------------
            # According to Nuscenes prediction benchmark
            # ------------------------
            if category in self.target_vehicle_category and attribute in self.target_vehicle_attribute:
                return True
            else:
                return False

        elif self.category_filtering_method == 2:
            # ------------------------
            # According to Trajectron++ (ECCV, 2020)
            # ------------------------
            ped_flag = False
            if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
                ped_flag = True

            veh_flag = False
            if 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                veh_flag = True

            return ped_flag or veh_flag

        else:
            print("[Error] method %d is not supported !!" % self.category_filtering_method)
            sys.exit(0)

    def __len__(self):
        return len(self.token)
