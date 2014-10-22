#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys; sys.path.append('../')

from xkin.gesture import GestureClassifier

QUIT = 113
CLEAR = 99
SAVE = 115
WAIT_TIME = 200
RADIUS = 5
COLOR = (255, 0, 0)


class TestModels():

    def __init__(self, src, width=640, height=480):
        self._width = width
        self._height = height
        self._win = 'test models'
        self._img = None
        self._seq = None
        self._src = src
        self._gesture = GestureClassifier()
        self._gesture.load(src)
        self._refresh()

    def run(self):
        cv2.namedWindow(self._win)
        cv2.moveWindow(self._win, 0, 0)
        cv2.setMouseCallback(self._win, self._on_mouse, 0)
        while True:
            cv2.imshow(self._win, self._img)
            key = cv2.waitKey(WAIT_TIME)
            if key == QUIT:
                break
            elif key == ord('c'):
                if self._seq:
                    print self._gesture.run(np.array(self._seq), True)
                    self._refresh()
            else:
                continue

    def _refresh(self):
        self._img = np.zeros((self._height, self._width, 3), np.uint8)
        self._seq = []

    def _on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            self._seq.append((x, y))
            cv2.circle(self._img, pt, RADIUS, COLOR, -1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test models')
    parser.add_argument('--src', required=True)
    args = parser.parse_args()

    TestModels(args.src).run()
