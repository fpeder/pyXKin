#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pickle

QUIT = 113
CLEAR = 99
SAVE = 115
WAIT_TIME = 250
RADIUS = 5
COLOR = (255, 0, 0)


class GenGestureProto():

    def __init__(self, id, nstate, dst, width=640, height=480):
        self._width = width
        self._height = height
        self._win = 'gen proto'
        self._img = None
        self._id = id
        self._nstate = nstate
        self._dst = dst
        self._seq = None

    def run(self):
        cv2.namedWindow(self._win)
        cv2.setMouseCallback(self._win, self._on_mouse, 0)

        self._init()

        while True:
            cv2.imshow(self._win, self._img)
            key_pressed = cv2.waitKey(WAIT_TIME)

            if key_pressed == QUIT:
                break

            elif key_pressed == CLEAR:
                self._init()

            elif key_pressed == SAVE:
                output = {'id':self._id, 'nstate': self._nstate,
                          'seq':np.array(self._seq)}
                pickle.dump(output, open(self._dst, 'wb'))
                break

            else:
                continue

    def _init(self):
        self._img = np.zeros((self._height, self._width, 3), np.uint8)
        self._seq = []

    def _on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            self._seq.append((x, y))
            cv2.circle(self._img, pt, RADIUS, COLOR, -1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='gen proto')
    parser.add_argument('--id', required=True)
    parser.add_argument('--nstate', required=True, type=int)
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()

    ggp = GenGestureProto(args.id, int(args.nstate), args.dst).run()
