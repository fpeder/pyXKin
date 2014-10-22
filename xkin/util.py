#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def draw_points(points, win_name=None):
    points = points.astype(np.uint16)
    color = (255, 0, 0)
    tmp = np.zeros((480, 640, 3), np.uint8)
    for p in points:
        cv2.circle(tmp, (p[0], p[1]), 5, color, -1)

    if win_name:
        cv2.imshow(win_name, tmp)
    else:
        cv2.imshow('points', tmp)
        cv2.waitKey(0)
