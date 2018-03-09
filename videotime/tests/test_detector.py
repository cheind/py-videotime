import os
import videotime as vt
import glob
import cv2
from datetime import datetime

ETCPATH = os.path.join(os.path.dirname(__file__),'..','..','etc')

def test_files():
    fnames = glob.glob(os.path.join(ETCPATH, 'test0', '*.png'))
    y = [
        '08:43:48.12',
        '08:43:49.16',
        '08:43:50.12',
        '08:43:51.16',
        '08:43:52.11',
        '08:43:53.15',
        '08:43:54.11',
        '08:43:55.15',
        '08:43:56.11',
        '08:18:28.67',
        '08:15:25.99',
        '08:15:27.03',
        '08:15:27.99',
        '08:15:29.03',
        '08:15:29.99',
        '08:15:31.03',
        '08:15:31.99',
        '08:07:21.51',
    ]

    d = vt.TimeDetector.load(os.path.join('videotime.bin'))

    for idx, f in enumerate(fnames):
        img = cv2.imread(f)
        assert y[idx] == datetime.strftime(d.detect(img)[0], '%H:%M:%S.%f')[:-4]