#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from sklearn import cluster


class HandDetector():

    def __init__(self, func):
        self._func = func

    def run(self, depth, depth8=None, rgb=None):
        hand = self._func.run(depth, depth8)
        mask = hand.copy()
        mask[mask > 0] = 255
        return (hand, mask)


class HandMSK():
    DOWNS = 100
    Q = 0.4

    def __init__(self, K=None):
        self._K = K

    def run(self, depth, depth8):
        X, idx = self._get_forground(depth, depth8)
        H = np.zeros(X.shape, np.uint8)
        Xfg = np.expand_dims(X[idx], axis=1)

        if self._K == None:
            self._get_num_cluster(Xfg)

        labels, cid = self._clustering(Xfg)

        idx2 = idx[labels == cid]
        H[idx2] = depth8.flatten()[idx2]
        H = H.reshape(depth.shape[0], depth.shape[1])
        return H

    def _get_num_cluster(self, X):
        D = X[::self.DOWNS]
        bw = cluster.estimate_bandwidth(D, quantile=self.Q)
        clf = cluster.MeanShift(bandwidth=bw).fit(D)
        K = clf.cluster_centers_.shape[0]
        self._K = K

    def _clustering(self, X):
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data=X.astype(np.float32), K=self._K,
            bestLabels=None, criteria=crit, attempts=1,
            flags=cv2.KMEANS_RANDOM_CENTERS)

        labels = labels.flatten()
        cluster_id = np.argsort(centers[:,0])[0]
        return (labels, cluster_id)

    def _get_forground(self, depth, depth8):
        X = depth.flatten()
        th, _ = cv2.threshold(depth8, 127, 255, cv2.THRESH_OTSU |
                              cv2.THRESH_BINARY)
        th = th / 255. * 1047
        idx = np.where(np.logical_and(X > 0, X < th))[0]
        return X, idx


class HandOtsu():

    def __init__(self):
        pass

    def run(self, depth, depth8=None):
        thresh = cv2.THRESH_OTSU | cv2.THRESH_BINARY
        th, otsu  = cv2.threshold(depth8, 127, 255, thresh)

        #asd = depth8.copy(); asd[asd > th] = 0
        #cv2.imshow('otsu', asd)

        X      = depth8[depth8 < th]; X = X[X > 0]
        th, _  = cv2.threshold(X, 127, 255, thresh)
        depth8[depth8 > th] = 0
        return depth8


class HandMeanShift():

    def __init__(self):
        pass

    def run(self, depth, depth8=None):
        X = depth[depth < self._get_otsu_threshold]
        X = X[X>0]
        X = X[::100]

        X = np.expand_dims(X, axis=1)
        bw = cluster.estimate_bandwidth(X, quantile=0.4)
        clf = cluster.MeanShift(bandwidth=bw)
        clf.fit(X)

        centers = clf.cluster_centers_.flatten()
        labels = clf.labels_.flatten()
        idx = np.argsort(centers)

        th = X[labels==idx[0]].max()
        depth[depth > th] = 0
        return depth

    def _get_otsu_threshold(depth):
        thresh = cv2.THRESH_OTSU | cv2.THRESH_BINARY
        th, _  = cv2.threshold(depth, 127, 255, thresh)
        return th


class HandDetectorNew():
    Q = 0.15

    def __init__(self, ncluster=None):
        self._ncluster = ncluster
        self._labels  = None
        self._centers = None
        self._hand    = None

    def run(self, depth):
        depthf = depth.flatten()
        hand   = np.zeros(depthf.shape)
        nz     = np.where(depthf > 0)[0]
        X      = np.expand_dims(depthf[nz], axis=1)

        if self._ncluster == None:
            bw = cluster.estimate_bandwidth(X, quantile=self.Q)
            cl = cluster.MeanShift(bandwidth=bw, min_bin_freq=10)
            cl.fit(X)

            self._ncluster = len(np.unique(cl.labels_))

            labels  = cl.labels_
            centers = cl.cluster_centers_

        else:
            X = X.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        10, 1.0)

            _, labels, centers = cv2.kmeans(
                data=X.astype(np.float32), K=self._ncluster, bestLabels=None,
                criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

            labels = labels.flatten()

        cluster_id = np.argsort(centers[:,0])[0]

        idx = nz[labels == cluster_id]
        hand[idx] = depthf[idx]
        hand = hand.reshape(depth.shape[0], depth.shape[1])

        self._hand = hand

        return hand

    def _get_hand(self):
        pass

    def binarize(self):
        if self._hand != None:
            hand = self._hand.copy()
            hand[hand > 0] = 255
            hand = hand.astype(np.uint8)
        else:
            hand = None

        return hand

    @property
    def ncluster(self):
        return self._ncluster


class HandDetectorOld():

    def __init__(self):
        self._hand       = None
        self._interval   = None
        self._hand_depth = None
        self._mean_depth = None

    def run(self, body):
        data             = self._get_data_from_image(body)
        self._inter      = self._get_hand_interval(data)
        self._hand_depth = self._isolate_hand(body, self._inter)
        self._hand_bin   = self._binarize(body, self._inter)
        self._depth      = (self._inter[0] + self._inter[1]) / 2.0
        return (self._hand_depth, self._hand_bin)

    def _isolate_hand(self, body, inter):
        hand = body.copy()
        hand[body >= inter[1]] = 0
        return hand

    def _binarize(self, body, inter):
        hand = body.copy()
        hand[hand > inter[1]] = 0
        hand[np.logical_and(body >= inter[0], body <= inter[1])] = 255
        return hand

    def _get_hand_interval(self, data):
        kmeans = cluster.KMeans(n_clusters=2, init=data[1]).fit(data[0])
        th1, th2 = kmeans.cluster_centers_
        return (data[1][0], th1)

    def _get_data_from_image(self, body):
        data = body[body != 0].copy().reshape(-1, 1)
        seed = np.array([[data.min()], [data.max()]])
        return (data, seed)


class HandCrop():
    INF = 75
    SUP = 180

    def __init__(self):
        pass

    def run(self, hand, contour, center, r):
        angle = self._get_angle(contour)
        hand  = self._crop(hand, center, r, angle)
        return hand

    def _crop(self, hand, center, r, angle):
        hand = hand.copy()
        if angle > self.INF and angle < self.SUP:
            hand[center[1] + r:, :] = 0
        else:
            hand[:, center[0] + r:] = 0

        return hand

    def _get_angle(self, contour):
        angle = cv2.fitEllipse(contour)[-1]
        return angle
