#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import sys; sys.path.append('../')

from gesture import GestureResample
from gesture import stft


class TestSTFT():

    def __init__(self, width=640, height=480):
        self._width = width
        self._height = height
        self._win = 'test models'
        self._img = None
        self._seq = None
        self._resample = GestureResample(32)
        self._refresh()

    def run(self):
        cv2.namedWindow(self._win)
        cv2.moveWindow(self._win, 0, 0)
        cv2.setMouseCallback(self._win, self._on_mouse, 0)

        while True:
            cv2.imshow(self._win, self._img)
            key = cv2.waitKey(100)
            if key == ord('q'):
                break

            elif key == ord('c'):
                if self._seq:
                    seq = np.array(self._seq)
                    seq = self._resample.run(seq, True)

                    x = stft(seq)
                    print x

                    self._refresh()

    def _refresh(self):
        self._img = np.zeros((self._height, self._width, 3), np.uint8)
        self._seq = []

    def _on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            self._seq.append((x, y))
            cv2.circle(self._img, pt, 5, 255, -1)


if __name__ == '__main__':
    TestSTFT().run()





