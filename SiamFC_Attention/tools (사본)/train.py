from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = '/home/airlab/PycharmProjects/pythonProject5/data/LaSOTBenchmark'
    seqs = LaSOT(root_dir=root_dir,subset='train')

    tracker = TrackerSiamFC(net_path='/home/airlab/PycharmProjects/pythonProject5/Siam-DW/pretrained/attention_pretrained1.pth')
    tracker.train_over(seqs)
