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
from tqdm import tqdm
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
        self.actor = SAC(num_inputs=3042)
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
        #self.actor.load_model('/home/airlab/PycharmProjects/pythonProject5/SiamET3/tools/policy.pth')
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

            # train parameters
            'epoch_num': 1000,
            'batch_size': 32,
            'num_workers': 32,
            'number_of_updates':100
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
    def update(self, img,evaluate=True):
        self.net.eval()
        # search images

        x = ops.crop_and_resize(
            img, self.center, self.x_sz * 1.5,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color)
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(2,0,1).unsqueeze(0).float()

        state=self.state_generator(x)
        actions=self.get_action(state,evaluate=evaluate)

        self.center+=np.array([actions[0],actions[1]])
        self.target_sz=self.target_sz+self.target_sz*actions[2]
        self.z_sz = self.z_sz + self.z_sz * actions[2]
        self.x_sz = self.x_sz + self.x_sz * actions[2]


        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return state,actions,box

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
                _,_,boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img,boxes[f,:])
                #ops.image_save(img, boxes[f, :],report_dir="image_result",image_name=self.n,number=f)
        self.n+=1
        return boxes, times

    @torch.enable_grad()
    def supervised_learning(self,seqs,val_seqs=None,save_dir='models',sequential_frames=20):

        #RL
        memory=mixed_replay_buffer_stochastic(100000,12345,tau=0.1)

        epi=list()
        epi_reward=0
        R_MAX=-1000000
        self.net.eval()

        #create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataloader
        dataset=Online_Sequence(seqs=seqs,transforms=None,sequential_frames=sequential_frames)
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

            for it,batch in tqdm(enumerate(dataloader)):

                epi_reward=0
                epi=list()
                transition=list()
                images=batch[0]
                anno=batch[1]
                mask=0
                l=0

                for f in range(sequential_frames):
                    img = np.array(images[f].squeeze())
                    ann = np.array(anno[f].squeeze())
                    l += 1
                    if f == 0:
                        self.init(img, ann)

                    if f==sequential_frames-1:
                        break

                    else:
                        state,actions,box=self.update(img,evaluate=False)
                        next_img = np.array(images[f+1].squeeze())
                        next_state=self.next_state_generator(next_img)
                        reward,mask=self.reward_calcul(box,ann)

                        state = state.squeeze(1).detach().cpu().numpy()
                        next_state = next_state.squeeze(1).detach().cpu().numpy()

                        transition = [state, actions, reward, next_state, mask]
                        memory.push(*transition)
                        epi_reward += reward
                        epi.append(transition)

                        if mask==1:
                            break
                if epi_reward > R_MAX:
                    R_MAX = epi_reward
                    for t in epi:
                        memory.hpush(*t)

                if len(memory) > self.cfg.batch_size:
                    for iter in range(self.cfg.number_of_updates):
                        self.actor.update_parameters(memory, batch_size=self.cfg.batch_size)



                if it%10==0:
                    print("epoch: ",epoch,"Accu Reward: ", epi_reward,"length: ",l)
                    torch.save(self.actor.policy.state_dict(), "policy.pth")

    @torch.enable_grad()
    def unsupervised_learning(self, seqs, val_seqs=None, save_dir='models', sequential_frames=20):

        # RL
        memory = mixed_replay_buffer_stochastic(100000, 12345, tau=0.1)

        epi = list()
        epi_reward = 0
        R_MAX = -1000000
        self.net.eval()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataloader
        dataset = Online_Sequence(seqs=seqs, transforms=None, sequential_frames=sequential_frames)
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

                for f in range(sequential_frames):
                    img = np.array(images[f].squeeze())
                    ann = np.array(anno[f].squeeze())
                    l += 1
                    if f == 0:
                        self.init(img, ann)

                    if f == sequential_frames - 1:
                        break

                    else:
                        state, actions, box = self.update(img, evaluate=False)
                        next_img = np.array(images[f + 1].squeeze())
                        next_state = self.next_state_generator(next_img)


                        state = state.squeeze(1).detach().cpu().numpy()
                        next_state = next_state.squeeze(1).detach().cpu().numpy()

                        transition = [state, actions, next_state]
                        epi.append(transition)

                        if f == sequential_frames - 2:
                            reward,mask=self.reward_calcul(box,ann)

                for t in epi:
                    memory.push(*[t[0],t[1],reward,t[2],0])



                if reward > R_MAX:
                    R_MAX = reward
                    for t in epi:
                        memory.hpush(*[t[0],t[1],reward,t[2],0])

                if len(memory) > self.cfg.batch_size:
                    for iter in range(self.cfg.number_of_updates):
                        self.actor.update_parameters(memory, batch_size=self.cfg.batch_size)

                if it % 10 == 0:
                    print("epoch: ", epoch, "Accu Reward: ", reward, "length: ", l)
                    torch.save(self.actor.policy.state_dict(), "policy.pth")


    def state_generator(self,x):
        # responses - state1
        response = self.net(self.kernel, x)
        response = response.squeeze(1).cpu().numpy()[0]
        response = cv2.resize(
            response, (51,51),
            interpolation=cv2.INTER_CUBIC)
        response=np.array([response])
        response = torch.from_numpy(response).to(
            self.device).permute(2,0,1).unsqueeze(0).float()
        response=nn.Flatten()(response)



        # feature_map of search images - state2
        feature_map=self.net.feature_extractor(x)
        feature_map=self.net.dimension_reduction(feature_map)
        feature_map=nn.Flatten()(feature_map)

        state=torch.cat([response,feature_map],1).to(self.device)

        return state

    @torch.no_grad()
    def next_state_generator(self,img):
        self.net.eval()
        # search images

        x = ops.crop_and_resize(
            img, self.center, self.x_sz * 1.5,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color)
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(2,0,1).unsqueeze(0).float()

        state=self.state_generator(x)
        return state

    def reward_calcul(self,box,ann):

        iou=rect_iou(box,ann)
        if iou>0.5:
            reward=iou
            mask=0
        else:
            reward=-1
            mask=1

        return reward,mask

    def get_action(self,state,evaluate):
        actions=self.actor.select_action(state,evaluate=evaluate)
        return actions