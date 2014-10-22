#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np

class Calib():

    def __init__(self, fn='../data/calib.pck'):
        (self._dc, #depth
         self._vc, #rgb
         self._T,
         self._R) = self._load(fn)

    def box_depth2rgb(self, box, depth):
        p1 = (box[0], box[1])
        z1 = depth[p1[1], p1[0]]

        p2 = (box[0] + box[2], box[1] + box[3])
        z2 = depth[p2[1], p2[0]]

        o1 = self.point_depth2rgb(p1, z1)
        o2 = self.point_depth2rgb(p2, z2)

        box = (o1, o2 - o1)
        box = (box[0][0], box[0][1], box[1][0], box[1][1])
        return box

    def point_depth2rgb(self, pt, z):
        t1 = np.array([pt[0], pt[1], 0], dtype=np.float)

        t1[0:2] = (t1[0:2] - self._dc[2:4]) * z / self._dc[0:2] + self._T[0:2]
        t1[2] = z + self._T[2]

        t2 = np.zeros(3, dtype=np.float)
        for i in range(3):
            t2[i] = (self._R[i, :] * t1).sum()

        out = (t2[0:2] * self._vc[0:2] / t2[2] + self._vc[2:4]).astype(np.int)
        return out

    def _load(self, fn):
        data = pickle.load(open(fn, 'rb'))
        return data['depth'], data['rgb'], data['T'], data['R']
