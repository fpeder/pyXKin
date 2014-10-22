#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import config
import numpy as np

from preprocess import PreProcess
from calib import Calib


class HandContourDetector():
    MEDIAN_DIM = config.MEDIAN_DIM
    MORPH_SHAPE_DIM = config.MORPH_SHAPE_DIM
    MORPH_NITER = config.MORPH_NITER
    MIN_CONTOUR_LEN = config.MIN_CONTOUR_LEN
    AR_LIM = 1.5

    def __init__(self):
        self._contour = None
        self._box = None

    def run(self, hand, rgb=False, box=None, hand_crop=None, depth=None):
        if rgb and box and hand_crop.any() and depth.any():
            phand = self._rgb_hand_seg(hand, box, hand_crop, depth)
        else:
            phand = PreProcess().median_smooth(hand, self.MEDIAN_DIM)

        tmp  = phand.copy()
        cont = self._get_largest_contour(phand)
        box  = cv2.boundingRect(cont)
        crop = self._crop_box(tmp, box)
        self._contour = cont
        self._box = box
        return (cont, box, crop)

    def _crop_box(self, img, box):
        crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].copy()
        return crop

    def _get_largest_contour(self, hand):
        contours = cv2.findContours(hand, cv2.cv.CV_RETR_EXTERNAL,
                                    cv2.cv.CV_CHAIN_APPROX_NONE)
        lengths = [cv2.arcLength(x, True) for x in contours[0]]

        def valid_aspect_ratio(contour):
            box = cv2.boundingRect(contour)
            notvalid = float(box[3])/box[2] > self.AR_LIM or \
                       float(box[2])/box[3] > self.AR_LIM
            return notvalid

        for i in reversed(np.argsort(lengths)):
            contour = contours[0][i]
            if valid_aspect_ratio(contour):
                break
        return contour

    def _rgb_hand_seg(self, hand, box, asd, depth):
        box = Calib().box_depth2rgb(box, depth)

        roi = self._crop_box(hand, box)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        roi = cv2.threshold(roi[:, :, 2], 0, 255, cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        asd = cv2.dilate(asd, kernel, iterations=3)
        asd = cv2.resize(asd, (roi.shape[1], roi.shape[0]),
                         interpolation=cv2.cv.CV_INTER_LINEAR)

        roi[asd == 0] = 0
        roi = cv2.medianBlur(roi, 5)
        return roi

    def draw(self):
        tmp = np.zeros((self._box[3], self._box[2], 3), np.uint8)
        cv2.drawContours(tmp, self._contour, -1, (0, 0, 255))
        cv2.imshow('contour', tmp)

    def not_valid(self):
        cond = len(self._contour) < self.MIN_CONTOUR_LEN
        return cond
