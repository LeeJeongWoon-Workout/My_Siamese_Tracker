from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path='/home/airlab/PycharmProjects/pythonProject5/siamfc-pytorch-master/pretrained/SiamFCRes22W.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir='/home/airlab/PycharmProjects/pythonProject5/data/OTB2015'
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker)
    e.report([tracker.name])
