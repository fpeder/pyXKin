#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import config

from kinect import Kinect
from body import BodyDetector
from hand import HandDetector, HandContourDetector
from pose import PoseClassifier, OpenCloseClassifier, MultiPoseClassifier


kinect  = Kinect()
body    = BodyDetector()
hand    = HandDetector()
contour = HandContourDetector()
#pose   = PoseClassifier(OpenCloseClassifier())
pose    = PoseClassifier(MultiPoseClassifier())


for (depth, depth8, rgb) in kinect.get_data():

    b = body.run(depth8)
    (h, _) = hand.run(b)

    #cv2.imshow('hand', h)

    (ccc, box, hc) = contour.run(h)

    if len(ccc) < 100:
        continue

    (ccc, _, _) = contour.run(rgb, True, box, hc, depth)

    p = pose.run(ccc)
    if p == -1:
        continue

    if kinect.stop(kinect.wait()):
        break
