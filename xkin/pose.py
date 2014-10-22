#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config
import numpy as np
import pickle

from features import ConvexityDefects, FourierDescriptors


class PoseClassifier():
    BUFFLEN = config.BUFFLEN
    NONE = config.HAND_NONE
    buffer = []

    def __init__(self, classifier=None):
        self._classifier = classifier

    def run(self, contour):
        if self._classifier:
            curr_pose = self._classifier.run(contour)

            if len(self.buffer) <= self.BUFFLEN:
                self.buffer.append(curr_pose)
                pose = self.NONE
            else:
                self.buffer.pop(0)
                self.buffer.append(curr_pose)
                pose = self._majority_voting()

            return pose

    def _majority_voting(self):
        buffer = np.array(self.buffer)
        votes = [len(np.where(buffer == x)[0]) for x in buffer]
        pose = np.argsort(votes)[-1]

        return buffer[pose]


class OpenCloseClassifier():
    DEFECTS_DEPTH_FACTOR = config.DEFECTS_DEPTH_FACTOR
    NUM_DEFECTS = config.NUM_DEFECTS
    OPEN = config.HAND_OPEN
    CLOSE = config.HAND_CLOSE
    NONE = config.HAND_NONE

    def __init__(self):
        self._cvxdefects = ConvexityDefects()

    def run(self, contour):
        (defects, box) = self._cvxdefects.run(contour)

        if not defects:
            return self.NONE

        if defects == None:
            return self.CLOSE

        if self._is_open(defects, box):
            pose = self.OPEN
        else:
            pose = self.CLOSE

        return pose

    def _is_open(self, defects, box):
        asd  = (box[2] * box[3]) / 2.0
        num  = defects.size
        mean = defects[:, :, -1].mean()

        c1 = mean >= (float(asd) / self.DEFECTS_DEPTH_FACTOR)
        c2 = num >= self.NUM_DEFECTS

        return (c1 and c2)


class MultiPoseClassifier():

    def __init__(self, src):
        self._fourier_desc = FourierDescriptors()
        self._models = self._load_models(src)

    def run(self, contour):
        desc = self._fourier_desc.run(contour)
        dist = np.zeros(len(self._models))

        for i, model in enumerate(self._models):
            dist[i] = model['model'].mahalanobis([desc])

        pose = self._models[np.argsort(dist)[0]]['id']

        return pose

    def _load_models(self, src):
        models = []
        for s in src:
            data = pickle.load(open(s, 'rb'))
            models.append({'id': data['id'], 'model': data['model']})

        return models
