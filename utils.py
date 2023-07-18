
import torch
from torch.utils.data import Dataset
from NuScenes_utils.train_data import DatasetCreator
from tqdm import tqdm
import numpy as np
import random
import argparse
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrainValDataLoader(Dataset):
    def __init__(self, args, validation=False):
        super().__init__()
        self.args = args
        self.parent_dir = Path(__file__).parents[0].resolve()
        self.dir = 'val_set' if validation else 'train_set'

        #  Create train-Val dataset if not exists
        if not os.path.exists(f'{self.parent_dir}/dataset_for_train/{self.dir}'):
            DatasetCreator(args).get_train_val()

        self.num_scene = len(os.listdir(f'{self.parent_dir}/dataset_for_train/{self.dir}'))
        print(f'Found {self.num_scene} scenes')

    def __len__(self):
        return self.num_scene

    def __getitem__(self, i):
        data = torch.load(f'{self.parent_dir}/dataset_for_train/{self.dir}/data_{i}.pth', map_location='cpu')
        obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, maps, num_neighbors, valid_neighbor, possible_paths, \
            lane_label = data
        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
            maps, torch.tensor(num_neighbors), torch.tensor(valid_neighbor), possible_paths, lane_label




class GetArgs:
    def getargs(self):
        p = argparse.ArgumentParser()

        p.add_argument('--model_name', type=str, default='HLS')

        p.add_argument('--model_mode', type=str, default='vehicle')

        p.add_argument('--exp_id', type=int, default=np.random.randint(1000, 9999,1))
        p.add_argument('--gpu_num', type=int, default=0)
        p.add_argument('--device', type=str, default='cuda:0')
        p.add_argument('--load_pretrained', type=int, default=0)
        p.add_argument('--start_epoch', type=int, default=0)
        p.add_argument('--multi_gpu', type=int, default=1)
        p.add_argument('--num_cores', type=int, default=8)

        p.add_argument('--dataset_path', type=str, default='dataset_raw')

        p.add_argument('--dataset_type', type=str, default='nuscenes')
        p.add_argument('--preprocess_trajectory', type=int, default=0)
        p.add_argument('--num_turn_scene_repeats', type=int, default=0)
        p.add_argument('--input_dim', type=int, default=2)
        p.add_argument('--scene_accept_prob', type=float, default=.33)

        p.add_argument('--past_horizon_seconds', type=float, default=2)
        p.add_argument('--future_horizon_seconds', type=float, default=6)
        p.add_argument('--target_sample_period', type=float, default=2)  # Hz
        p.add_argument('--min_past_horizon_seconds', type=float, default=1.5)
        p.add_argument('--min_future_horizon_seconds', type=float, default=3)
        p.add_argument('--val_ratio', type=float, default=.05)
        p.add_argument('--max_num_agents', type=int, default=100)
        p.add_argument('--min_num_agents', type=int, default=2)
        p.add_argument('--stop_agents_remove_prob', type=float, default=0)
        p.add_argument('--limit_range_change_prob', type=float, default=.0)

        p.add_argument('--category_filtering_method', type=int, default=0)

        p.add_argument('--num_train_scenes', type=int, default=10000)
        p.add_argument('--num_val_scenes', type=int, default=10000)
        p.add_argument('--num_test_scenes', type=int, default=10000)

        # Hyperparameter
        p.add_argument('--num_epochs', type=int, default=100)
        p.add_argument('--batch_size', type=int, default=8)
        p.add_argument('--best_k', type=int, default=5)
        p.add_argument('--learning_rate', type=float, default=.0001)
        p.add_argument('--min_learning_rate', type=float, default=.00001)
        p.add_argument('--learning_rate_cnn', type=float, default=.00005)
        p.add_argument('--grad_clip', type=float, default=0.0)
        p.add_argument('--n_cycle', type=int, default=4)
        p.add_argument('--warmup_ratio', type=float, default=.5)

        p.add_argument('--alpha', type=float, default=1.0)
        p.add_argument('--beta', type=float, default=.5)
        p.add_argument('--gamma', type=float, default=.01)
        p.add_argument('--kappa', type=float, default=1.0)

        p.add_argument('--valid_step', type=int, default=1)
        p.add_argument('--save_every', type=int, default=3)
        p.add_argument('--max_num_chkpts', type=int, default=5)

        p.add_argument('--apply_cyclic_schedule', type=int, default=0)
        p.add_argument('--separate_lr_for_cnn', type=int, default=0)
        p.add_argument('--apply_lr_scheduling', type=int, default=0)

        p.add_argument('--limit_range', type=int, default=30)

        # Model Argument
        p.add_argument('--is_random_path_order', type=int, default=1)
        p.add_argument('--is_train_dis', type=int, default=1)

        p.add_argument('--path_resol', type=float, default=1.)
        p.add_argument('--max_path_len_forward', type=float, default=80)
        p.add_argument('--max_path_len_backward', type=float, default=10)
        p.add_argument('--ngh_dist_thr', type=float, default=5)

        p.add_argument('--num_max_paths', type=int, default=10)
        p.add_argument('--lane_feat_dim', type=int, default=64)

        p.add_argument('--pos_emb_dim', type=int, default=16)
        p.add_argument('--traj_enc_h_dim', type=int, default=16)
        p.add_argument('--traj_dec_h_dim', type=int, default=128)

        p.add_argument('--gan_prior_prob', type=float, default=.5)
        p.add_argument('--z_dim', type=int, default=16)
        args = p.parse_args()
        return args


def check_dir(dirs):
    if dirs != '' and not os.path.exists(dirs):
        os.makedirs(dirs)

def seq_collate_typeA(data):

    obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
    map, num_neighbors, valid_neighbor, possible_path, lane_label = zip(*data)

    _len = [objs for objs in num_neighbors]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_ta_cat = torch.cat(obs_traj_ta, dim=1)
    future_traj_ta_cat = torch.cat(future_traj_ta, dim=1)
    obs_traj_ngh_cat = torch.cat(obs_traj_ngh, dim=1)
    future_traj_ngh_cat = torch.cat(future_traj_ngh, dim=1)

    map_cat = torch.cat(map, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    possible_path_cat = torch.cat(possible_path, dim=1)
    lane_label_cat = torch.cat(lane_label, dim=0)

    return tuple([obs_traj_ta_cat, future_traj_ta_cat, obs_traj_ngh_cat, future_traj_ngh_cat,
                  map_cat, seq_start_end, torch.tensor(valid_neighbor), possible_path_cat, lane_label_cat])
