#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import cv
import numpy as np
import sys; sys.path.append('../')

from kinect import Kinect
from body import BodyDetector
from hand import HandDetector, HandOtsu, HandMSK
from contour import HandContourDetector
from palm import PalmDetector

NBINS = 2**11 - 1


class TestHand():

    def __init__(self):
        self._kinect  = Kinect()
        self._body    = BodyDetector()
        self._hand    = HandDetector(HandOtsu())
        self._contour = HandContourDetector()
        self._palm    = PalmDetector()

    def run(self):
        for (depth, depth8, rgb) in self._kinect.get_data():

            cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            cv2.imshow('depth', depth8)

            hand, mask = self._hand.run(depth, depth8)
            (_, _, crop) = self._contour.run(mask)

            if crop == None:
                continue

            cv2.imshow('hand', crop)

            hand = self._palm.run(hand, crop)
            if hand == None:
                continue

            cv2.imshow('hand final', hand)

            cv2.waitKey(10)


    def _crop(self, img, box):
        crop = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        return crop


if __name__ == '__main__':
    TestHand().run()
