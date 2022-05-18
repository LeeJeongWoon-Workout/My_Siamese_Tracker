from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = '/home/airlab/PycharmProjects/pythonProject5/data/LaSOTBenchmark'
    seqs = LaSOT(root_dir)
    net_path="/home/airlab/PycharmProjects/pythonProject5/SiamET4/pretrained/attention_pretrained1.pth"
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.online_train(seqs)
