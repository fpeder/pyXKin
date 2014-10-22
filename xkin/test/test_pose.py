#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from kinect import Kinect
from body import BodyDetector
from hand import HandDetector, HandContourDetector

from pose import PoseClassifier, MultiPoseClassifier


class TestPose():

    def __init__(self, src):
        self._src     = src
        self._kinect  = Kinect()
        self._body    = BodyDetector()
        self._hand    = HandDetector()
        self._contour = HandContourDetector()
        self._pose    = PoseClassifier(MultiPoseClassifier(src))

    def run(self):
        for (depth, depth8, rgb) in self._kinect.get_data():
            contour = self._get_hand_contour(depth8, depth, rgb)

            if contour.any():
                self._contour.draw()
                print self._pose.run(contour)

            cv2.waitKey(5)

    def _get_hand_contour(self, depth8, depth, rgb):
        body            = self._body.run(depth8)
        (hand, _)       = self._hand.run(body)
        (cont, box, hc) = self._contour.run(hand)

        if self._contour.not_valid():
            return np.array([])

        (cont, _, _) = self._contour.run(rgb, True, box, hc, depth)

        return cont


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='gen proto')
    parser.add_argument('--model', required=True, nargs='+')
    args = parser.parse_args()

    TestPose(args.model).run()
