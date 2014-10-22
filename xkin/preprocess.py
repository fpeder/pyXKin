#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2


class PreProcess():

    def __init__(self):
        pass

    def median_smooth(self, img, size):
        img = cv2.medianBlur(img, size)

        return img

    def morph_smooth(self, img, elem_size, niter):
        elem = cv2.getStructuringElement(cv2.cv.CV_SHAPE_ELLIPSE, elem_size)
        img  = cv2.morphologyEx(img, cv2.MORPH_OPEN, elem, niter)
        img  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, elem, niter)

        return img
