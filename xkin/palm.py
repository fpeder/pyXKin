#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class PalmDetector():
    THRESH  = 0.9
    SIZE    = (29, 29)
    VAR     = 150/0.3
    MIN_LEN = 50
    COVER   = 0.95

    def __init__(self):
        self._min_pos = None

    def run(self, handd, handb, contour=None):
        self._min_pos = self._get_min_depth_pos(handd)
        density = self._get_density_map(handb)
        blobs   = self._get_blobs(density)
        center  = self._get_starting_point(blobs)
        if not center:
            return None

        (p, c, r) = self._fit_palm_cirle(density, handb, center)

        tmp = cv2.cvtColor(density, cv2.COLOR_GRAY2BGR)
        cv2.circle(tmp, center, 5, (0,0,255))
        cv2.circle(tmp, center, r, (0,255,0))
        cv2.imshow('palm', tmp)

        hand = self._palm_cut(handb, p, c, r)
        return hand

    def _palm_cut(self, hand, palm, c, r):
        hand = hand.copy()
        hand[c[1] + r + 1:, :] = 0
        return hand

    def _fit_palm_cirle(self, density, handb, center):
        r = 1
        rprev = 0
        nin = 5

        while r - rprev > 0:
            while nin > self.COVER:
                mask = np.zeros(handb.shape, np.uint8)
                cv2.circle(mask, center, r, 255, -1)
                palm = mask * handb
                nin = len(palm[palm > 0]) / (r**2 * np.pi)
                r = r + 5
            rprev = r
            #center = self._update_center(density, palm)
        return (palm, center, r)

    def _update_center(self, density, palm):
        tmp = density * palm
        max_pos = np.where(tmp == tmp[tmp > 0].max())
        max_pos = (max_pos[1][0], max_pos[0][0])
        return max_pos

    def _get_min_depth_pos(self, handd):
        tmp = handd
        min_pos = np.where(tmp == tmp[tmp > 0].min())
        min_pos = (min_pos[1][0], min_pos[0][0])
        return min_pos

    def _get_density_map(self, hand):
        hand = cv2.GaussianBlur(hand, self.SIZE, self.VAR)
        return hand

    def _get_blobs(self, hand):
        blobs = hand.copy()
        blobs[hand <  self.THRESH * hand.max()] = 0
        blobs[hand >= self.THRESH * hand.max()] = 255
        return blobs

    def _get_starting_point(self, blobs):
        contour = cv2.findContours(blobs, cv2.cv.CV_RETR_EXTERNAL,
                                   cv2.cv.CV_CHAIN_APPROX_NONE)

        def _point_dist(p1, p2):
            p1 = np.array(p1)
            p2 = np.array(p2)
            d = np.sqrt(np.sum((p1 - p2)**2))
            return d

        def _get_centroid(contour):
            mom  = cv2.moments(contour)
            cent = (int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00']))
            return cent

        cent = [_get_centroid(cont) for cont in contour[0] if
                len(cont) >= self.MIN_LEN]

        if cent:
            dist = [_point_dist(self._min_pos, pt) for pt in cent]
            pt = cent[np.argsort(dist)[0]]
        else:
            pt = []

        return pt
