#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import sys;
sys.path.append('../')

from xkin.gesture import GestureTraining

seq = [pickle.load(open(x,'rb')) for x in sys.argv[1:]]
seq = [list(x.itervalues()) for x in seq]

gt = GestureTraining()
gt.run(seq)
gt.save('gesture5.pck')
