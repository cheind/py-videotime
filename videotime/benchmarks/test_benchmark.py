import os
import videotime as vt
import glob
import cv2

ETCPATH = os.path.join(os.path.dirname(__file__),'..','..','etc')

d = vt.TimeDetector.load(os.path.join('videotime.bin'))
f = glob.glob(os.path.join(ETCPATH, 'test0', '*.png'))[0]
img = cv2.imread(f)

def run_detector():
    d.detect(img)

def test_benchmark_detector(benchmark):    
    benchmark(run_detector)
