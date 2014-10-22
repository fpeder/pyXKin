#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import cv2

from kinect import Kinect
from body import BodyDetector
from hand import HandDetector, HandContourDetector
#from features import FourierDescriptors
#from sklearn.covariance import EmpiricalCovariance
from features import GaborDescriptors

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
        self._feature  = GaborDescriptors(4, 4)

    def run(self):
        warmup = True
        train = []
        model = pickle.load(open('svm.pck', 'rb'))

        for (depth, depth8, rgb) in self._kinect.get_data():
            body      = self._body.run(depth8)
            (hand, _) = self._hand.run(body)
            (cont, box, crop) = self._contour.run(hand)

            hand = hand[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

            cv2.imshow('hand', hand)
            key = cv2.waitKey(2)

            #if warmup:
            #    if key == GO:
            #        warmup = False
            #    continue

            #if key != 97:
            #    continue

            feature = self._feature.run(hand)
            print model.predict(feature)
            #train.append(feature)

            #if len(train) == self._nsamples:
            #    self._save(train)
            #    break

            #cv2.waitKey(2)

    def _save(self, train):
        data = np.array(train)
        labels = self._id * np.ones(len(train), np.int)
        output = {'labels': labels, 'data': data}
        pickle.dump(output, open(self._dst, 'wb'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='gen proto')
    parser.add_argument('--id', required=True, type=int)
    parser.add_argument('--nsamples', required=True, type=int)
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()

    TrainPose(args.id, args.nsamples, args.dst).run()
