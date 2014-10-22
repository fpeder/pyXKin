#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys; sys.path.append('../')

from xkin.gesture import GestureFeatures, GestureResample

WAIT_TIME = 100
RADIUS = 5
COLOR = (255, 0, 0)

class TestFeatures():

    def __init__(self, width=640, height=480):
        self._width = width
        self._height = height
        self._win = 'test features'
        self._img = None
        self._seq = None
        self._feature = GestureFeatures()
        self._resample = GestureResample(16)
        self._refresh()

    def run(self):
        cv2.namedWindow(self._win)
        cv2.moveWindow(self._win, 0, 0)
        cv2.setMouseCallback(self._win, self._on_mouse, 0)

        while True:
            cv2.imshow(self._win, self._img)
            key = cv2.waitKey(WAIT_TIME)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._refresh()
            elif key == ord('c'):
                if self._seq:
                    seq = np.array(self._seq)
                    seq  = self._resample.run(seq, True)
                    self._draw_seq(seq)
                    f1, f2, f3, f4 = self._feature.run(seq)
                    print f4
                    self._refresh()

                print 'asd'
            else:
                continue

    def _refresh(self):
        self._img = np.zeros((self._height, self._width, 3), np.uint8)
        self._seq = []

    def _on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            self._seq.append(pt)
            cv2.circle(self._img, pt, RADIUS, COLOR, -1)

    def _draw_seq(self, seq):
        tmp = np.zeros((self._height, self._width, 3), np.uint8)
        for pt in seq.astype(np.int).tolist():
            cv2.circle(tmp, tuple(pt), 5, (255,0,0), -1)
        cv2.imshow(self._win, tmp)
        cv2.waitKey(200)


if __name__ == '__main__':
    #import argparse
    #parser = argparse.ArgumentParser(description='test models')
    #parser.add_argument('--src', required=True)
    #args = parser.parse_args()

    TestFeatures().run()
