from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = '/home/airlab/PycharmProjects/pythonProject5/data/GOT-10K'
    seqs = GOT10k(root_dir, subset='train')
    net_path="/home/airlab/PycharmProjects/pythonProject5/SiamET3/pretrained/SiamFCRes22W.pth"
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.online_train(seqs)
