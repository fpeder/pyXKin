#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import config
import freenect


WAIT_TIME = config.WAIT_TIME


class Kinect(object):

    def __init__(self):
        pass

    def get_data(self):
        while True:
            (depth, _) = freenect.sync_get_depth()
            (rgb  , _) = freenect.sync_get_video()

            depth8 = self._pretty_depth(depth)
            yield depth, depth8, rgb

    def _pretty_depth(self, depth):
        depth = depth.copy()
        np.clip(depth, 0, 2**10 - 1, depth)
        depth >>= 2
        depth = depth.astype(np.uint8)
        return depth

    def wait(self):
        key = cv2.waitKey(WAIT_TIME)
        return key

    def stop(self, key):
        bool = key == 27 or key == 113
        return bool


if __name__ == '__main__':
    kinect = Kinect()
    for (_, d, r) in kinect.get_data():

        cv2.imshow('depth', d)

        if kinect.stop(cv2.waitKey(config.WAIT_TIME)):
            break
