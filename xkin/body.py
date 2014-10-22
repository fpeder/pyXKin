#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import config

NBINS = config.NBINS
AREA_MIN = 0.5 #config.AREA_MIN
ASD = 0.95


class BodyDetector():
    WINDOW_NAME = 'body'

    def __init__(self):
        self._depth    = None
        self._hist     = None
        self._interval = None
        self._body     = None

    def run(self, img):
        self._depth    = img
        self._hist     = self._get_depth_histogram(img)
        self._interval = self._get_body_depth_interval(self._hist)
        self._interval = self._refine_interval()
        self._body     = self._isolate_body(self._depth, self._interval)
        return self._body

    def _get_depth_histogram(self, img):
        hist = cv2.calcHist([img], [0], None, [NBINS], [0, NBINS])
        hist[-1] = 0
        hist = self._normalize(hist)
        return hist

    def _isolate_body(self, img, interval):
        dst = img
        dst[img < interval[0]] = 0
        dst[img > interval[1]] = 0
        return dst

    def _get_body_depth_interval(self, hist):
        nz_idx = np.where(hist > 0)[0]
        LNZ = len(nz_idx)
        area = 0.0
        i = 0

        while i < LNZ and area < AREA_MIN:
            j = i + 1
            if j >= LNZ:
                j = LNZ - 1
                break

            while nz_idx[j] == nz_idx[i] + j - i:
                j += 1
                if j >= LNZ:
                    j = LNZ - 1
                    break

            area = hist[nz_idx[i:j]].sum()
            interval = (nz_idx[i], nz_idx[j])

            i = j + 1
            if i >= LNZ:
                i = LNZ - 1
                break

        return interval


    def _refine_interval(self):
        h = self._hist[self._interval[0] : self._interval[1]]
        h = self._normalize(h)
        sum = 0
        i = 1

        while sum < ASD:
            sum = h[0:i].sum()
            i += 1
        return (self._interval[0], self._interval[0] + i)

    def _normalize(self, x):
        xn = x / x.sum()
        return xn

    def show(self):
        if self._body.any() and self._depth.any():
            img = np.hstack((self._depth, self._body))
            str = 'th1=%d th2=%d' %(self._interval[0], self._interval[1])
            cv2.imshow(str, img)


if __name__ == '__main__':
    from kinect import Kinect

    kinect = Kinect()
    body = BodyDetector()
    

    for (d, _) in kinect.get_data():

        b = body.run(d)

        #body.show()

        if kinect.stop(kinect.wait()):
            break

        #import sys; sys.stdin.read(1)
