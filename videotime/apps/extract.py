if __name__ == '__main__':
    import numpy as np
    import cv2
    import argparse
    import glob
    import os
    import videotime as vt

    parser = argparse.ArgumentParser(description='Train time detector')
    parser.add_argument('model', help='Model file')
    parser.add_argument('input', help='Directory containing images and image pattern, or path to video')    
    parser.add_argument('--verbose', action='store_true')  
    args = parser.parse_args()

    detector = vt.TimeDetector.load(args.model)
    cap = cv2.VideoCapture(args.input)

    timestamps = []
    while True:
        b, img = cap.read()        
        if not b:
            break

        dtime, prob = detector.detect(img)
        timestamps.append(dtime)
        print(dtime)