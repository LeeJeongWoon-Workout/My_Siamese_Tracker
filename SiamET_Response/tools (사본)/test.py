from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path='//home/airlab/PycharmProjects/pythonProject5/SiamET2/pretrained/SiamFCRes22W.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir='/home/airlab/PycharmProjects/pythonProject5/data/OTB2013'
    e = ExperimentOTB(root_dir, version=2013)
    e.run(tracker,visualize=True)
    e.report([tracker.name])
