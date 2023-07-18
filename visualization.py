import torch

from nuscenes.nuscenes import NuScenes
from NuScenes_utils.map import Map

from NuScenes_utils.visualization import Visualizer
from preprocessing_utils.functions import *
from NuScenes_utils.train_data import DatasetCreator  # load_datasetloader, load_solvers
import os
from utils import GetArgs
from trainer import Trainer
from NuScenes_utils.train_data import DatasetSaver


class Visualization:
    def __init__(self):
        self.local_args = self.get_args()
        self.global_args = GetArgs().getargs()
        self.global_args.exp_id = 1000
        self.global_args.best_k = self.local_args.best_k

        self.data_helper = DatasetSaver(args=self.global_args, isTrain=False)

        self.parent_dir = Path(__file__).parents[0].resolve()
        if not os.path.exists(f'{self.parent_dir}/dataset_for_train/test_set'):
            DatasetCreator(args=self.global_args).get_test()

        self.trainer = Trainer(args=self.global_args)
        # self.trainer.load_param()

        pass

    def fit(self):

        self.global_args.best_k = self.local_args.best_k
        self.global_args.batch_size = 4
        self.global_args.limit_range = 200

        # evaluation setting
        t_skip = self.local_args.t_skip
        obs_len = int(self.global_args.past_horizon_seconds * self.global_args.target_sample_period)
        pred_len = int(self.global_args.future_horizon_seconds * self.global_args.target_sample_period)
        self.global_args.best_k = self.local_args.best_k
        self.global_args.batch_size = 1
        obs_len_ = obs_len

        # sub-sample trajs
        target_index_obs = np.array([-1 * _ for _ in range(0, obs_len, t_skip)])[::-1] + (obs_len - 1)
        target_index_pred = np.array([_ for _ in range(t_skip - 1, pred_len, t_skip)])

        obs_len = len(target_index_obs)
        pred_len = len(target_index_pred)

        # scene range
        map_size = self.local_args.map_size
        x_range = (-1 * self.local_args.scene_range, self.local_args.scene_range)
        y_range = (-1 * self.local_args.scene_range, self.local_args.scene_range)
        z_range = (-3, 2)

        # visualizer
        vs = Visualizer(args=self.global_args, map=self.data_helper.map, x_range=x_range, y_range=y_range,
                        z_range=z_range, map_size=map_size, obs_len=obs_len, pred_len=pred_len)

        scene_list = os.listdir(f'{self.parent_dir}/dataset_for_train/test_set')
        self.local_args.end_frm_idx = len(scene_list)
        target_sa = create_target_scene_agent_list()
        target_scenes = np.unique(target_sa[:, 0]).tolist()

        for idx, current_scene_list in enumerate(scene_list):
            if self.local_args.is_target_only:
                if not idx == target_scenes:
                    continue
            data = torch.load(f'{self.parent_dir}/dataset_for_train/test_set/{current_scene_list}', map_location='cpu')

            obs_traj, future_traj, pred_trajs, agent_ids, scene, valid_scene_flag = self.trainer.testing(data)

            obs_traj = obs_traj.detach().cpu().numpy()
            future_traj = future_traj.detach().cpu().numpy()
            pred_trajs = pred_trajs.detach().cpu().numpy()

            agent_samples, scene = scene

            obs_traj, future_traj, pred_trajs = self.data_helper.convert_to_egocentric(obs_traj, future_traj,
                                                                                       pred_trajs, agent_ids,
                                                                                       agent_samples)

            obs_traj_valid = obs_traj[target_index_obs, :, :]
            future_traj_valid = future_traj[target_index_pred, :, :]
            overall_traj = np.concatenate([obs_traj_valid, future_traj_valid], axis=0)
            pred_trajs_valid = pred_trajs[:, target_index_pred, :, :]

            num_agents = agent_ids.shape[0]
            for a in range(num_agents):
                if self.local_args.is_target_only and not chk_if_target(target_sa, idx, a):
                    continue

                a_token = scene.id_2_token_lookup[agent_ids[a]]
                agent = scene.agent_dict[a_token]

                #  draw point cloud topivew
                lidar_now_token = scene.lidar_token_seq[obs_len_ - 1]
                fig, ax = vs.topview_pc(lidar_now_token, IsEmpty=True)

                # draw hdmap
                scene_location = scene.city_name
                ax = vs.topview_hdmap(ax, lidar_now_token, scene_location, x_range, y_range, map_size, agent=agent,
                                      IsAgentOnly=True, BestMatchLaneOnly=False)

                # draw bbox
                for n in range(num_agents):
                    n_token = scene.id_2_token_lookup[agent_ids[n]]
                    neighbor = scene.agent_dict[n_token]

                    if n != a:
                        xy = neighbor.trajectory[obs_len_ - 1, 1:3].reshape(1, 2)
                        ax = vs.topview_bbox(ax, neighbor, xy, (0.5, 0.5, 0.5))
                xy = agent.trajectory[obs_len_ - 1, 1:3].reshape(1, 2)
                ax = vs.topview_bbox(ax, agent, xy, (1, 0, 0))

                # draw traj
                gt_traj = overall_traj[:, a, :]
                for k in range(self.local_args.best_k):
                    est_traj = pred_trajs_valid[k, :, a, :]
                    ax = vs.topview_trajectory(ax, gt_traj, est_traj)
                plt.axis([0, map_size, 0, map_size])

                img = vs.fig_to_nparray(fig, ax)
                text = '[Scene %d - Agent %d]' % (idx, a)
                cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

                cv2.imshow('', img)
                cv2.waitKey(0)
                plt.close()
                pass

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--exp_id', type=int, default=300)
        parser.add_argument('--gpu_num', type=int, default=0)
        parser.add_argument('--model_name', type=str, default='HLS')
        parser.add_argument('--start_frm_idx', type=int, default=0)
        parser.add_argument('--end_frm_idx', type=int, default=1)
        parser.add_argument('--best_k', type=int, default=15)
        parser.add_argument('--map_size', type=int, default=1024)
        parser.add_argument('--t_skip', type=int, default=1)
        parser.add_argument('--scene_range', type=float, default=70)
        parser.add_argument('--is_target_only', type=bool, default=False)
        parser.add_argument('--target_scene_idx', type=int, default=1)

        args = parser.parse_args()
        return args


def create_target_scene_agent_list():
    target_scene_agent = [[0, 1]]
    return np.array(target_scene_agent)


def chk_if_target(target_sa, sid, aid):
    target_s = target_sa[target_sa[:, 0] == sid]
    if aid in target_s[:, 1].tolist():
        return True
    else:
        return False


if __name__ == '__main__':
    Visualization().fit()
