import torch
# import numpy as np

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import HLS_Model, Discriminator
import torch.nn.functional as F
import random
import sys
import numpy as np


class Trainer:
    def __init__(self, args):
        self.args = args
        self.best_k = args.best_k
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.kappa = args.kappa
        self.batch_size = args.batch_size
        self.n_cycle = args.n_cycle
        self.warmup_ratio = args.warmup_ratio
        self.num_max_paths = args.num_max_paths
        self.is_train_dis = args.is_train_dis

        self.num_batches = int(args.num_train_scenes / args.batch_size)
        self.total_num_iteration = args.num_epochs * self.num_batches

        self.iter = 0
        self.l2_losses = 0
        self.kld_losses = 0
        self.bce_losses = 0
        self.g_losses = 0
        self.d_losses = 0
        self.current_ADE = 0
        self.best_ADE = 1e5
        self.cur_lr = args.learning_rate
        self.min_lr = args.min_learning_rate

        self.model = HLS_Model(args=args).to(self.args.device)
        self.discriminator = Discriminator(args=args).to(self.args.device)

        self.opt = optim.Adam(list(self.discriminator.parameters()) + list(self.model.parameters()),
                              lr=args.learning_rate)
        self.apply_kld_scheduling = args.apply_cyclic_schedule
        self.apply_lr_scheduling = args.apply_lr_scheduling

        self.kld_weight_scheduler = self.create_kld_weight_scheduler()
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.9999)

    def train(self, train_iter, val_iter):
        for epoch in range(self.args.num_epochs):
            self.init_loss_tracker()
            for data in tqdm(train_iter, desc=f'Training Iter: {epoch + 1} is progressing'):
                self.gen(data)
                self.disc(data)
            self.normalize_loss_tracker()
            self.eval(val_iter)
            print(f'Epoch {epoch} || ADE: {self.current_ADE} || Best ADE: {self.best_ADE :.4f}')
            pass

    def gen(self, data):
        obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor, \
            possible_lane, lane_label = data

        obs_traj = obs_traj.to(self.args.device)
        future_traj = future_traj.to(self.args.device)
        obs_traj_ngh = obs_traj_ngh.to(self.args.device)
        future_traj_ngh = future_traj_ngh.to(self.args.device)
        seq_start_end = seq_start_end.to(self.args.device)
        valid_neighbor = valid_neighbor.to(self.args.device)
        possible_lane = possible_lane.to(self.args.device)
        lane_label = lane_label.to(self.args.device)

        feat_map = torch.zeros(1).to(self.args.device)

        self.model.train()
        self.opt.zero_grad()

        pred_trajs, pred_offsets, means0, log_vars0, means1, log_vars1, logits, lane_context \
            = self.model(obs_traj[:, :, :4],
                         future_traj[:, :, :4],
                         obs_traj_ngh[:, :, :4],
                         future_traj_ngh[:, :, :4],
                         feat_map,
                         seq_start_end,
                         valid_neighbor,
                         possible_lane,
                         lane_label)
        pred_trajs = pred_trajs.permute(0, 2, 1, 3).contiguous()

        #  L2 Loss
        l2 = torch.zeros(1).to(obs_traj.to(self.args.device))
        for i in range(self.best_k):
            l2 += self.l2_loss(pred_trajs[i], future_traj[:, :, 2:4].to(self.args.device))
        l2 = l2 / float(self.best_k)
        self.l2_losses += l2.item()

        #  KLD Loss
        kld = self.kld_loss(means0, log_vars0, means1, log_vars1)
        self.kld_losses += kld.item()
        if self.iter > len(self.kld_weight_scheduler) - 1:
            self.iter = len(self.kld_weight_scheduler) - 1

        # bce loss
        bce = self.cross_entropy_loss(logits, lane_label)
        self.bce_losses += bce.item()

        # final loss
        loss = l2 + (self.alpha * bce) + \
               (self.beta * self.kld_weight_scheduler[self.iter] * kld)

        # back-propagation
        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        self.opt.step()

    def disc(self, data):
        obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor, \
            possible_lane, lane_label = data

        obs_traj = obs_traj.to(self.args.device)
        future_traj = future_traj.to(self.args.device)
        obs_traj_ngh = obs_traj_ngh.to(self.args.device)
        future_traj_ngh = future_traj_ngh.to(self.args.device)
        seq_start_end = seq_start_end.to(self.args.device)
        valid_neighbor = valid_neighbor.to(self.args.device)
        possible_lane = possible_lane.to(self.args.device)
        lane_label = lane_label.to(self.args.device)

        feat_map = torch.zeros(1).to(self.args.device)

        # find corresponding lane positions for GT future trajectories
        seq_len, batch, _ = future_traj.size()
        future_traj_np = future_traj[:, :, 2:4]
        ref_lanes, corr_lane_pos_gt = [], []
        for b in range(batch):
            idx = torch.argwhere(lane_label[b, :] == 1)[0][0]
            cur_lanes = possible_lane[:, b * self.num_max_paths:(b + 1) * self.num_max_paths, :]

            ref_lane = cur_lanes[:, idx, :]
            ref_lanes.append(ref_lane)

            corr_pos = self.find_corr_lane_position(future_traj_np[:, b, :], ref_lane).reshape(seq_len, 1, 2)
            corr_lane_pos_gt.append(corr_pos)
        corr_lane_pos_gt = torch.cat(corr_lane_pos_gt, dim=1)
        corr_lane_pos_gt = corr_lane_pos_gt.to(obs_traj)

        # process map data

        self.model.train()
        self.discriminator.train()
        self.opt.zero_grad()

        # predict future trajectory
        pred_trajs, lane_context = self.model(obs_traj[:, :, :4],
                                              future_traj[:, :, :4],
                                              obs_traj_ngh[:, :, :4],
                                              future_traj_ngh[:, :, :4],
                                              feat_map,
                                              seq_start_end,
                                              valid_neighbor,
                                              possible_lane,
                                              lane_label,
                                              best_k=1)
        pred_trajs = pred_trajs.permute(0, 2, 1, 3).contiguous()

        # find corresponding lane positions for predicted future trajectories
        pred_trajs_np = pred_trajs
        corr_lane_pos = []
        for b in range(batch):
            corr_pos = self.find_corr_lane_position(pred_trajs_np[0][:, b, :], ref_lanes[b]).reshape(seq_len, 1, 2)
            corr_lane_pos.append(corr_pos)
        corr_lane_pos = torch.cat(corr_lane_pos, dim=1)
        corr_lane_pos = corr_lane_pos.to(obs_traj)

        # d-loss
        scr_real = self.discriminator(future_traj[:, :, 2:4].to(self.args.device),
                                      corr_lane_pos_gt.to(self.args.device), lane_context)
        scr_fake_for_D = self.discriminator(pred_trajs[0].detach(), corr_lane_pos.to(self.args.device), lane_context)
        d_loss = self.gan_d_loss(scr_real, scr_fake_for_D)
        self.d_losses += d_loss.item()

        # g loss
        scr_fake_for_G = self.discriminator(pred_trajs[0], corr_lane_pos.to(self.args.device), lane_context)
        g_loss = self.gan_g_loss(scr_fake_for_G)
        self.g_losses += g_loss.item()

        # final loss
        loss = d_loss + (self.gamma * g_loss)

        # back-propagation
        loss.backward()
        self.opt.step()

    def eval(self, val_iter):
        '''
                    obs_traj : seq_len x batch x 4 (speed, heading, x, y)
                    future_traj : seq_len x batch x 4 (speed, heading, x, y)
                    obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
                    future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
                    map : batch x dim x h x w
                    seq_start_end : batch x 2
                    valid_neighbor : batch
                    possible_lanes : seq_len x (batch x num_max_paths) x 2
        '''

        self.model.eval()
        self.discriminator.eval()

        ADE = []
        num_samples = int(self.args.num_val_scenes / self.args.batch_size) * self.args.batch_size
        for data in val_iter:
            obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, \
                maps, seq_start_end, valid_neighbor, possible_lane, lane_label = data

            seq_start_end = torch.LongTensor(seq_start_end)

            obs_traj = obs_traj.to(self.args.device)
            future_traj = future_traj.to(self.args.device)
            obs_traj_ngh = obs_traj_ngh.to(self.args.device)
            future_traj_ngh = future_traj_ngh.to(self.args.device)
            seq_start_end = seq_start_end.to(self.args.device)
            valid_neighbor = valid_neighbor.to(self.args.device)
            possible_lane = possible_lane.to(self.args.device)
            lane_label = lane_label.to(self.args.device)

            # process map data
            feat_map = torch.zeros(1).to(self.args.device)

            # predict future trajectory
            pred_trajs = self.model.inference(obs_traj[:, :, :4],
                                              future_traj[:, :, :4],
                                              obs_traj_ngh[:, :, :4],
                                              future_traj_ngh[:, :, :4],
                                              feat_map,
                                              seq_start_end,
                                              valid_neighbor,
                                              possible_lane,
                                              lane_label).permute(0, 2, 1, 3)
            for k in range(self.best_k):
                err = torch.sqrt(torch.sum((pred_trajs[k] - future_traj[:, :, 2:4]) ** 2, dim=2))
                ADE.append(err)
        self.current_ADE = torch.mean(torch.stack(ADE))
        if self.best_ADE > self.current_ADE:
            self.best_ADE = self.current_ADE
            self.save_param()

    def l2_loss(self, pred_traj, pred_traj_gt):

        seq_len, batch, _ = pred_traj.size()
        loss = (pred_traj_gt - pred_traj) ** 2

        return torch.sum(loss) / (seq_len * batch)

    def kld_loss(self, mean1, log_var1, mean2, log_var2):

        '''
        KLD = -0.5 * (log(var1/var2) - (var1 + (mu1 - mu2)^2)/var2 + 1 )
            = -0.5 * (A - B + 1)

        A = log(var1) - log(var2)
        B = (var1 + (mu1 - mu2)^2) / var2

        prior ~ N(mean2, var2)
        posterior ~ N(mean1, var1)
        '''

        A = log_var1 - log_var2
        B = log_var1.exp() + (mean1 - mean2).pow(2)
        kld = -0.5 * (A - B.div(log_var2.exp() + 1e-10) + 1)

        return torch.sum(kld, dim=1).mean()

    def cross_entropy_loss(self, logit, target):
        return F.binary_cross_entropy_with_logits(logit, target, reduction='mean') / logit.size(0)

    def create_kld_weight_scheduler(self):
        scheduler = self.frange_cycle_linear(self.total_num_iteration, n_cycle=self.n_cycle, ratio=self.warmup_ratio)

        if self.args.apply_cyclic_schedule == 1:
            return scheduler
        else:
            return torch.ones_like(scheduler)

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
        L = torch.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L

    def find_corr_lane_position(self, traj, lane):

        """
        traj : seq_len x 2
        lane : seq_len x 2
        """
        lane = lane.to(self.args.device)
        seq_len = traj.shape[0]
        if torch.count_nonzero(torch.isnan(lane)) == 0:
            corr_lane_pos = []
            for t in range(seq_len):
                cur_pos = traj[t, :].reshape(1, 2).to(self.args.device)
                error = torch.sum(torch.abs(lane - cur_pos), dim=1)
                minidx = torch.argmin(error)
                corr_pos = lane[minidx, :].reshape(1, 2)
                corr_lane_pos.append(corr_pos)
            return torch.cat(corr_lane_pos, dim=0).to(self.args.device)

        else:
            return traj.to(self.args.device)

    def gan_d_loss(self, scores_real, scores_fake):
        """
        Input:
        - scores_real: Tensor of shape (N,) giving scores for real samples
        - scores_fake: Tensor of shape (N,) giving scores for fake samples

        Output:
        - loss: Tensor of shape (,) giving GAN discriminator loss
        """

        y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.0)
        y_fake = torch.zeros_like(scores_fake)
        loss_real = self.bce_loss(scores_real, y_real)
        loss_fake = self.bce_loss(scores_fake, y_fake)
        loss = loss_real + loss_fake

        return loss

    def gan_g_loss(self, scores_fake):
        """
        Input:
        - scores_fake: Tensor of shape (N,) containing scores for fake samples

        Output:
        - loss: Tensor of shape (,) giving GAN generator loss
        """

        y_fake = torch.ones_like(scores_fake)
        return self.bce_loss(scores_fake, y_fake)

    def bce_loss(self, inputs, target):
        """
        Numerically stable version of the binary cross-entropy loss function.
        As per https://github.com/pytorch/pytorch/issues/751
        See the TensorFlow docs for a derivation of this formula:
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        Input:
        - input: PyTorch Tensor of shape (N, ) giving scores.
        - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

        Output:
        - A PyTorch Tensor containing the mean BCE loss over the minibatch of
          input data.
        """
        neg_abs = -inputs.abs()
        loss = inputs.clamp(min=0) - inputs * target + (1 + neg_abs.exp()).log()
        return loss.mean()

    def init_loss_tracker(self):
        self.l2_losses = 0
        self.kld_losses = 0
        self.bce_losses = 0
        self.g_losses = 0
        self.d_losses = 0

    def normalize_loss_tracker(self):
        self.l2_losses /= self.num_batches
        self.kld_losses /= self.num_batches
        self.bce_losses /= self.num_batches
        self.g_losses /= self.num_batches
        self.d_losses /= self.num_batches

    def save_param(self):
        file_name = f'weight/nuscene_{self.args.exp_id}.pth'
        check_point = {
            'model': self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'opt': self.opt.state_dict(),
            'ADE': self.best_ADE}
        torch.save(check_point, file_name)

    def load_param(self):
        file_name = f'weight/nuscene_{self.args.exp_id}.pth'
        param = torch.load(file_name, map_location=self.args.device)
        self.model.load_state_dict(param['model'])

    def testing(self, data):
        """
        obs_traj : seq_len x batch x 4 (speed, heading, x, y)
        future_traj : seq_len x batch x 4 (speed, heading, x, y)
        obs_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        future_traj_ngh : seq_len x num_total_neighbors x 4 (speed, heading, x, y)
        map : batch x dim x h x w
        seq_start_end : batch x 2
        valid_neighbor : batch
        possible_lanes : seq_len x (batch x num_max_paths) x 2
        """
        self.model.eval()

        batch = 2

        _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _seq_start_end, _valid_neighbor, \
            _possible_lane, _lane_label, _agent_ids, _agent_samples, _scene = data

        pred_trajs = []
        num_agents = _obs_traj.size(1)

        for start in range(0, num_agents, batch):
            end = start + batch
            if end > num_agents:
                end = num_agents

            obs_traj = _obs_traj[:, start:end, :4]
            future_traj = _future_traj[:, start:end, :4]
            possible_lane = _possible_lane[:, start * self.num_max_paths:end * self.num_max_paths, :]
            valid_neighbor = torch.asarray(_valid_neighbor[start:end])
            lane_label = _lane_label[start:end]

            seq_start_end_copy = _seq_start_end[start:end]
            obs_traj_ngh, future_traj_ngh, seq_start_end = [], [], []

            pivot = seq_start_end_copy[0, 0]

            for i in range(seq_start_end_copy.shape[0]):
                _start = seq_start_end_copy[i, 0].item()
                _end = seq_start_end_copy[i, 1].item()

                seq_start_end.append(np.array([_start - pivot, _end - pivot]).reshape(1, 2))
                obs_traj_ngh.append(_obs_traj_ngh[:, _start:_end, :4])
                future_traj_ngh.append(_future_traj_ngh[:, _start:_end, :4])

            seq_start_end = torch.asarray(np.concatenate(seq_start_end, axis=0))
            obs_traj_ngh = torch.cat(obs_traj_ngh, dim=1)
            future_traj_ngh = torch.cat(future_traj_ngh, dim=1)

            remain = end - start
            num_padd = batch - (end - start)
            if num_padd > 0:
                obs_traj = self.padd(obs_traj, num_padd)
                future_traj = self.padd(future_traj, num_padd)
                obs_traj_ngh = self.padd(obs_traj_ngh, num_padd)
                future_traj_ngh = self.padd(future_traj_ngh, num_padd)
                possible_lane = self.padd(possible_lane, num_padd * self.num_max_paths)
                lane_label = self.padd(lane_label, num_padd, dim=0)

                for i in range(num_padd):
                    seq_start_end = torch.cat([seq_start_end, seq_start_end[-1].reshape(1, 2)], dim=0)
                    valid_neighbor = torch.cat([valid_neighbor, torch.tensor([False])])

            # process map data
            obs_traj = obs_traj.to(self.args.device)
            future_traj = future_traj.to(self.args.device)
            obs_traj_ngh = obs_traj_ngh.to(self.args.device)
            future_traj_ngh = future_traj_ngh.to(self.args.device)
            seq_start_end = seq_start_end.to(self.args.device)
            valid_neighbor = valid_neighbor.to(self.args.device)
            possible_lane = possible_lane.to(self.args.device)
            lane_label = lane_label.to(self.args.device)

            feat_map = torch.zeros(1).to(self.args.device)

            pred_trajs_mini = self.model.inference(obs_traj[:, :, :4],
                                                   future_traj[:, :, :4],
                                                   obs_traj_ngh[:, :, :4],
                                                   future_traj_ngh[:, :, :4],
                                                   feat_map,
                                                   seq_start_end,
                                                   valid_neighbor,
                                                   possible_lane,
                                                   lane_label).permute(0, 2, 1, 3)

            pred_trajs.append(pred_trajs_mini[:, :, :remain, :])

        pred_trajs = torch.cat(pred_trajs, dim=2)

        return _obs_traj[:, :, 2:], _future_traj[:, :, 2:], pred_trajs, _agent_ids, \
            (_agent_samples, _scene), True

    def padd(self, tensor, num_padd, dim=1):

        num_dim = len(list(tensor.size()))

        if num_dim == 3:
            d0, d1, d2 = tensor.size()
            if dim == 0:
                padd = torch.zeros(size=(num_padd, d1, d2)).to(tensor)
            elif dim == 1:
                padd = torch.zeros(size=(d0, num_padd, d2)).to(tensor)
            else:
                padd = torch.zeros(size=(d0, d1, num_padd)).to(tensor)

        elif num_dim == 2:
            d0, d1 = tensor.size()
            if dim == 0:
                padd = torch.zeros(size=(num_padd, d1)).to(tensor)
            elif dim == 1:
                padd = torch.zeros(size=(d0, num_padd)).to(tensor)

        else:
            sys.exit('dim %d is outside of the tensor dimension' % dim)

        return torch.cat((tensor, padd), dim=dim)
