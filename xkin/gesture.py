#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import config
import numpy as np
import pickle

from sklearn.hmm import GaussianHMM
#from sklearn.cluster import KMeans


class GestureClassifier():

    def __init__(self):
        self._resample = GestureResample(32)
        self._features = GestureFeatures()
        self._models = None

    def run(self, seq, debug=False):
        seq = self._resample.run(seq, upsample=True)
        f1, f2 = self._features.run(seq)
        seq = np.vstack((f1[:, 1], f2)).T

        scores = [model['model'].score(seq) for model in self._models]
        if debug:
            print seq
            print scores
            print '-------'

        gesture = self._models[np.argsort(scores)[-1]]['id']
        return gesture

    def load(self, src):
        data = pickle.load(open(src, 'rb'))
        self._models = data['models']


class GestureTraining():

    def __init__(self, K=4, covar='diag'):
        self._training = GestureTrainingSetGen()
        self._feature  = GestureFeatures()
        self._K = K
        self._covar = covar
        self._vq = None
        self._modesl = None

    def run(self, protos):
        models = []
        for nstate, label, seq in protos:
            train  = self._training.run(seq)
            f1, f2 = self._feature.run(train, True)

            o = np.vstack((f1[:,1], f2)).T

            (start, trans) = self.init_left_right_model(nstate)
            clf = GaussianHMM(n_components=nstate, covariance_type=self._covar,
                              transmat=trans, startprob=start)
            clf.fit(np.array([o]))
            models.append({'id':label, 'model':clf})

        self._models = models
        return models

    def init_left_right_model(self, nstate, type='left-right'):
        norm  = lambda x: x/x.sum()
        trans = np.zeros((nstate, nstate))
        start = norm(np.random.rand(nstate))
        for i in range(nstate):
            trans[i, i:] = norm(np.random.rand(nstate-i))
        return (start, trans)

    def save(self, dst):
        if self._models:
            data = {'models': self._models}
            pickle.dump(data, open(dst, 'wb'))


class GestureFeatures():

    def __init__(self, nangle=config.NUM_ANGLES, N=4):
        self._nangles = nangle
        self._N = N

    def run(self, seq, train=False):
        if train:
            (p3, p1) = self._get_angles(seq[0])
            #p2 = self._get_stft(seq[0])
            #p3 = self._get_ang_variation(p3)
            p4 = self._get_integral_ang(p1)
            for s in seq[1:]:
                t3, t1 = self._get_angles(s)
                #p1 = np.vstack((p1, t1))
                #p3 = np.vstack((t3, self._get_ang_variation(s)))
                #p2 = np.vstack((p2, self._get_stft(s)))
                p1 = np.vstack((p1, t1))
                p4 = np.hstack((p4, self._get_integral_ang(t1)))
        else:
            (p3, p1) = self._get_angles(seq)
            #p2 = self._get_stft(seq)
            #p3 = self._get_ang_variation(p3)
            p4 = self._get_integral_ang(p1)
        return [p1, p4]

    def _get_angles(self, seq):

        def cart2pol(seq):
            seq = np.diff(seq, axis=0)
            out = np.zeros(seq.shape, np.float32)
            out[:, 0] = np.sqrt(seq[:, 0]**2 + seq[:, 1]**2)
            out[:, 1] = np.arctan2(seq[:, 1], seq[:, 0]) * 180./np.pi
            return out

        def quantiz(angle):
            angle = (angle / (360 / self._nangles)).astype(np.int)
            return angle

        seq  = cart2pol(seq)
        qseq = quantiz(seq)
        return (seq, qseq)

    def _get_ang_variation(self, x):
        N = 2
        ang = x[:, 1]
        x = np.hstack((ang[0]*np.ones(N), ang, ang[-1]*np.ones(N)))
        y = np.zeros(len(ang))
        for i in range(N, len(x) - N):
            tmp = np.abs(np.diff(x[i-N: i+N])).sum()/(2*N-1)
            y[i-N] = tmp
        return y

    def _get_integral_ang(self, x):
        ang = x[:, 1]
        y = np.zeros(len(ang))
        for i in range(len(y)):
            y[i] = np.diff(ang[0:i]).sum()
        y[-1] = y[-2]
        return y

    def _get_stft(self, x1):
        N = self._N
        x = np.vstack((x1[0] * np.ones((N, 2)), x1, x1[-1] * np.ones((N, 2))))
        y = np.zeros((len(x1) - 1 , 2*N-2))
        for i in range(N, len(x) - N - 1):
            tmp = x[i-N: i+N]
            tmp = tmp[:, 0] + 1j * tmp[:, 1]
            tmp = np.abs(np.fft.fft(tmp))
            y[i-N, :] = tmp[2:]/tmp[1] #np.abs(np.fft.fft(tmp))[1:]
        return y


class GestureTrainingSetGen():

    def __init__(self, nseq=50, xvar=config.XVAR, yvar=config.YVAR):
        self._nseq = nseq
        self._var = (15, 15)
        self._resample = GestureResample(32)

    def run(self, seq):
        train = []
        for i in range(self._nseq):
            seq = self._resample.upsample(seq)
            noisy_seq = self._add_awgn_noise(seq)
            noisy_seq = self._resample.run(noisy_seq)
            train.append(noisy_seq)
        return train

    def _add_awgn_noise(self, seq):
        noise = np.random.rand(seq.shape[0], seq.shape[1])
        noise[:, 0] = self._var[0] * noise[:, 0]
        noise[:, 1] = self._var[1] * noise[:, 1]
        out = seq + noise
        return out.astype(np.float32)


class GestureResample():

    def __init__(self, n=32):
        self._n = n

    def run(self, points, upsample=False):
        if upsample:
            points = self.upsample(points)
        I = cv2.arcLength(points.astype(np.float32), False) / (len(points)-1)
        new_points = points[0]
        D = 0
        for i in range(1, len(points)):
            d = self._dist(points[i], points[i-1])
            if D + d >= I:
                q = points[i-1] + ((I - D)/d) * (points[i] - points[i-1])
                new_points = np.vstack((new_points, q.astype(np.int32)))
                points[i, :] = q
                D = 0
            else:
                D = D + d
        return new_points

    def upsample(self, seq):
        seq = seq.astype(np.uint16)
        out = np.zeros((self._n, 2))
        out[:, 0] = cv2.resize(seq[:, 0], (1, self._n)).flatten()
        out[:, 1] = cv2.resize(seq[:, 1], (1, self._n)).flatten()
        return out

    def _dist(self, p1, p2):
        d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return d


if __name__ == '__main__':
    pass
    #import sys
    #from util import draw_points

    # gc = GestureClassifier()
    # for f in sys.argv[1:]:
    #     proto = pickle.load(open(f, 'rb'))
    #     gc.train(proto)
    # gc.save('gesture.pck')

    # points = pickle.load(open(sys.argv[1], 'rb'))['seq']
    # draw_points(points)
    # points = gp.upsample(points.astype(np.uint16), 32)
    # draw_points(points)
    # points = gp.resample(points)
    # draw_points(points)

    # seq = [pickle.load(open(x,'rb')) for x in sys.argv[1:]]
    # seq = [list(x.itervalues()) for x in seq]

    # gt = GestureTraining()
    # gt.run(seq)
    # gt.save('gesture2.pck')
