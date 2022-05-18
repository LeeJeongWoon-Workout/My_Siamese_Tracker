from __future__ import absolute_import, division

import numpy as np
import cv2
from torch.utils.data import Dataset
import random

__all__ = ['Online_Sequence']

class Online_Sequence(Dataset):


    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1):
        super(Online_Sequence, self).__init__()
        self.seqs = seqs
        #seqs : 훈련 데이터 video 숫
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq  # 应该是每一帧都给出一个
        # shuffle     沿着第一个axis打乱数组的顺序，内容不变，return None
        # permutation 跟 shuffle 效果一样，只不过返回一个变量，自身不变
        self.indices = np.random.permutation(len(seqs))
        # getattr 用于 获取属性值
        self.return_meta = getattr(seqs, 'return_meta', False)  # ？？？？？

    def __getitem__(self, index):

        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations

        if self.return_meta:
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None

        # filter out noisy frames
        val_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios)

        if len(val_indices)<50:
            index=np.random.choice(len(self))
            return self.__getitem__(index)

        start=random.randint(0,len(val_indices)-50)
        val_indices=val_indices[start:start+50]



        bb=list()
        bb_anno=list()
        for idx in val_indices:

            f=cv2.imread(img_files[idx],cv2.IMREAD_COLOR)
            f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            box_f=anno[idx]
            bb.append(f)
            bb_anno.append(box_f)



        return (bb,bb_anno)


    def __len__(self):
        return len(self.indices)

    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)

        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices


