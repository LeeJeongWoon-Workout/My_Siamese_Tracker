from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from tqdm import tqdm
from . import ops
from .losses import BalancedLoss
from .datasets import *
from .transforms import SiamFCTransforms
from .utils import *
from .model import *
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from D2RL_SAC import *
from mixed_replay_buffer_stochastic import *

__all__ = ['TrackerSiamFC']


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamET_only_response', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.actor = SAC(num_inputs=2601*3)
        # setup model
        self.net = SiamFCRes22W()
        # load checkpoint if provided
        if net_path is not None:
            load_pretrain(self.net, net_path)
        self.net = self.net.cuda()
        # set to evaluation mode
        model = resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]), nn.Flatten())
        self.feature_extractor = model.cuda()
        self.actor.load_model('/home/airlab/PycharmProjects/pythonProject5/SiamET3/tools/policy.pth')
        self.n=0
    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 3,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = z

    @torch.no_grad()
    def update(self, img):
        self.net.eval()
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        samples = x
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        responses = self.net(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        for i in range(3):
            responses[i] -= responses[i].min()
            responses[i] /= responses[i].sum() + 1e-16
            responses[i] = (1 - self.cfg.window_influence) * responses[i] + \
                       self.cfg.window_influence * self.hann_window

        loc = np.unravel_index(responses[scale_id].argmax(), responses[scale_id].shape)

        sample1 = torch.from_numpy(responses[0])
        sample2 = torch.from_numpy(responses[1])
        sample3 = torch.from_numpy(responses[2])

        s1=torch.flatten(sample1).unsqueeze(dim=0).cuda()
        s2 = torch.flatten(sample2).unsqueeze(dim=0).cuda()
        s3 = torch.flatten(sample3).unsqueeze(dim=0).cuda()

        state = torch.cat([s1, s2,s3], dim=1).float()

        # state: torch.Size([1, 2601X3])

        actions = self.actor.select_action(state, evaluate=True)
        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image
        self.center+=np.array([actions[0],actions[1]])


        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        self.target_sz = self.target_sz + self.target_sz * actions[2]
        self.z_sz = self.z_sz + self.z_sz * actions[2]
        self.x_sz = self.x_sz + self.x_sz * actions[2]
        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img,boxes[f,:])
                #ops.image_save(img, boxes[f, :],report_dir="image_result",image_name=self.n,number=f)
        self.n+=1
        return boxes, times

    def train_update(self, img, evaluate=False):
        self.net.eval()
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        samples = x
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        responses = self.net(self.kernel, x)
        responses = responses.squeeze(1).detach().cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        for i in range(3):
            responses[i] -= responses[i].min()
            responses[i] /= responses[i].sum() + 1e-16
            responses[i] = (1 - self.cfg.window_influence) * responses[i] + \
                           self.cfg.window_influence * self.hann_window

        loc = np.unravel_index(responses[scale_id].argmax(), responses[scale_id].shape)

        sample1 = torch.from_numpy(responses[0])
        sample2 = torch.from_numpy(responses[1])
        sample3 = torch.from_numpy(responses[2])

        s1 = torch.flatten(sample1).unsqueeze(dim=0).cuda()
        s2 = torch.flatten(sample2).unsqueeze(dim=0).cuda()
        s3 = torch.flatten(sample3).unsqueeze(dim=0).cuda()

        state = torch.cat([s1, s2, s3], dim=1).float()
        # state: torch.Size([1, 2601X3])

        actions = self.actor.select_action(state, evaluate=True)
        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image
        self.center+=np.array([actions[0],actions[1]])


        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        self.target_sz = self.target_sz + self.target_sz * actions[2]
        self.z_sz = self.z_sz + self.z_sz * actions[2]
        self.x_sz = self.x_sz + self.x_sz * actions[2]

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])


        return state, actions, box

    def next_state(self, img):
        self.net.eval()
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        samples = x
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        responses = self.net(self.kernel, x)
        responses = responses.squeeze(1).detach().cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        for i in range(3):
            responses[i] -= responses[i].min()
            responses[i] /= responses[i].sum() + 1e-16
            responses[i] = (1 - self.cfg.window_influence) * responses[i] + \
                           self.cfg.window_influence * self.hann_window

        loc = np.unravel_index(responses[scale_id].argmax(), responses[scale_id].shape)

        sample1 = torch.from_numpy(responses[0])
        sample2 = torch.from_numpy(responses[1])
        sample3 = torch.from_numpy(responses[2])

        s1 = torch.flatten(sample1).unsqueeze(dim=0).cuda()
        s2 = torch.flatten(sample2).unsqueeze(dim=0).cuda()
        s3 = torch.flatten(sample3).unsqueeze(dim=0).cuda()

        state = torch.cat([s1, s2, s3], dim=1).float()

        # state: torch.Size([1, 2601X3])

        return state

    @torch.enable_grad()
    def online_train(self, seqs, val_seqs=None, save_dir='models'):

        # RL
        memory = mixed_replay_buffer_stochastic(1000000, 12345)

        epi = list()
        epi_reward = 0
        R_MAX = -1000000
        self.net.eval()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = Online_Sequence(seqs=seqs, transforms=None)
        # setup dataloader

        dataloader = DataLoader(
            dataset,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
            drop_last=True,
            batch_size=1
        )
        accu_reward = 0
        # loop over epochs

        for epoch in range(self.cfg.epoch_num):

            for it, batch in tqdm(enumerate(dataloader)):
                epi_reward = 0
                epi = list()
                transition = list()
                images = batch[0]
                anno = batch[1]
                mask = 0
                l = 0
                for i in range(1):
                    for f in range(50):
                        img = np.array(images[f].squeeze())
                        ann = np.array(anno[f].squeeze())
                        l += 1
                        if f == 0:
                            self.init(img, ann)

                        if f == 49:
                            break

                        else:
                            state, action, box = self.train_update(img, evaluate=True)

                            next_img = np.array(images[f + 1].squeeze())
                            next_state = self.next_state(next_img)

                            iou = rect_iou(box, ann)

                            state = state.detach().cpu().numpy()
                            next_state = next_state.detach().cpu().numpy()
                            ops.show_image(img, box)
                            if iou > 0.5:
                                reward = iou
                                mask = 0
                            else:
                                reward = -1
                                mask = 1

                            transition = [state, action, reward, next_state, mask]
                            memory.push(*transition)
                            if len(memory) > self.cfg.batch_size:
                                self.actor.update_parameters(memory, batch_size=self.cfg.batch_size)
                            epi_reward += reward
                            epi.append(transition)
                            if mask==1:
                                break
                    if epi_reward > R_MAX:
                        R_MAX = epi_reward
                        for t in epi:
                            memory.hpush(*t)

                    print("Accu Reward: ", epi_reward,"length: ",l)
                    torch.save(self.actor.policy.state_dict(), "policy.pth")