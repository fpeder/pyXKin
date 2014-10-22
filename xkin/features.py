#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config
import numpy as np
import cv2

from skimage.filter import gabor_kernel


class ConvexityDefects():
    POLY_APPROX_PRECISION = config.POLY_APPROX_PRECISION
    MIN_HULL_POINTS = config.MIN_HULL_POINTS

    def __init__(self):
        pass

    def run(self, contour):
        contour = self._contour_approx(contour)
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < self.MIN_HULL_POINTS:
            return []

        defects = cv2.convexityDefects(contour, hull)
        box = cv2.boundingRect(contour)
        return (defects, box)

    def _contour_approx(self, contour):
        contour = cv2.approxPolyDP(contour, self.POLY_APPROX_PRECISION, True)
        return contour


class LBPDescriptors():

    def __init__(self, r=1, n=8):
        self._r = r
        self._n = n

    def run(self, img):
        from mahotas.features import lbp
        coeff = lbp(img, self._r, self._n)
        return coeff


class HOGDescriptors():

    def __init__(self, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1)):
        self._orientations = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block

    def run(self, img):
        from skimage.feature import hog
        img = cv2.resize(img, (128,128))
        fg = hog(img, orientations=self._orientations,
                 pixels_per_cell=self._pixels_per_cell,
                 cells_per_block=self._cells_per_block)
        return fg


class FourierDescriptors():

    def __init__(self, ndesc=config.FD_NUM, contour_len=config.CONTOUR_LEN):
        self._ndesc = ndesc
        self._contour_len = contour_len

    def run(self, contour):
        contour = self._contour_resampling(contour)
        coeff   = self._get_coefficients(contour)
        return coeff

    def _get_coefficients(self, contour):
        x = np.array([np.complex(x, y) for x, y in contour[:, 0]])
        X = np.abs(np.fft.fft(x))[1:self._ndesc + 2]
        coeff = [X[i]/X[0] for i in range(1, self._ndesc + 1)]
        return coeff

    def _contour_resampling(self, contour):
        tmp = contour.astype(np.uint16)
        resampled = np.zeros((self._contour_len, 1, 2), np.uint16)
        for i in range(1):
            resampled[:, :, i] = cv2.resize(tmp[:, :, i],
                                            (1, self._contour_len),
                                            interpolation = cv2.INTER_CUBIC)
        return resampled


class GaborDescriptors():

    def __init__(self, nscale=4, ntheta=4, bw=1, size=(128,128)):
        self._nscale   = nscale
        self._ntheta   = ntheta
        self._size     = size
        self._filtered = None

    def run(self, img):
        img   = cv2.resize(img, self._size) if self._size else img
        freq  = [1./2**i for i in range(1, self._nscale + 1)]
        theta = [i*np.pi/self._ntheta for i in range(self._ntheta)]

        filtered = [self._gabor_filter(img, f, t) for f in reversed(freq)
                    for t in theta]

        self._filtered = filtered
        features = self._gaussian_avarage(filtered)
        return features

    def _gabor_filter(self, img, f, t):
        kernel = gabor_kernel(f, t)
        output = cv2.filter2D(img, -1, kernel.real) + 1j * \
                 cv2.filter2D(img, -1, kernel.imag)
        return output

    def _gaussian_avarage(self, filtered):
        x, y = np.meshgrid(range(self._size[1]), range(self._size[0]))
        f = lambda x,y,i,j: np.exp(-0.5 * ((x - (16*i-8))**2 + \
                                           (y - (16*j-8))**2)/8**2)

        features = []
        for k, img in enumerate(filtered):
            for j in range(1, 8+1):
                for i in range(1, 8+1):
                    features.append((f(x, y, i, j) * img).sum())
        return features

    def show(self):
        import pylab as pl

        for i in range(len(self._filtered)):
            pl.subplot(self._nscale, self._ntheta, i+1)
            pl.imshow(self._filtered[i].real)
        pl.show()


if __name__ == '__main__':
    img = cv2.imread('lena.jpg', 0)
    features = GaborDescriptors(4, 4).run(img)
