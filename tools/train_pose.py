#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import cv2

from kinect import Kinect
from body import BodyDetector
from hand import HandDetector, HandContourDetector
from features import FourierDescriptors
from sklearn.covariance import EmpiricalCovariance

GO = 103


class TrainPose():

    def __init__(self, id, nsamples, dst):
        self._id       = id
        self._nsamples = nsamples
        self._dst      = dst
        self._kinect   = Kinect()
        self._body     = BodyDetector()
        self._hand     = HandDetector()
        self._contour  = HandContourDetector()
        self._fdesc    = FourierDescriptors()
        self._train    = []

    def run(self):
        warmup = True
        for (depth8, depth, rgb) in self._kinect.get_data():
            contour = self._get_hand_contour(depth8, depth, rgb)
            if not contour:
                continue

            self._contour.draw()

            if warmup:
                key = cv2.waitKey(5)
                if key == GO:
                    warmup = False
                continue

            fd = self._fdesc.run(contour)
            self._train.append(fd)

            if len(self._train) == self._nsamples:
                self._save()
                break

            cv2.waitKey(5)

    def _get_hand_contour(self, depth8, depth, rgb):
        body = self._body.run(depth8)
        (hand, _) = self._hand.run(body)
        (cont, box, hc) = self._contour.run(hand)

        if self._contour.not_valid():
            return []

        (cont, _, _) = self._contour.run(rgb, True, box, hc, depth)

        return cont

    def _save(self):
        data = np.array(self._train)
        model = EmpiricalCovariance().fit(np.array(self._train))
        output = {'id': self._id, 'data': data,  'model': model}
        pickle.dump(output, open(self._dst, 'wb'))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='gen proto')
    parser.add_argument('--id', required=True, type=int)
    parser.add_argument('--nsamples', required=True, type=int)
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()

    TrainPose(args.id, args.nsamples, args.dst).run()
