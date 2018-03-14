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
    parser.add_argument('--out', help='Name of resulting file', default='timestamps.txt')
    args = parser.parse_args()

    detector = vt.TimeDetector.load(args.model)
    cap = cv2.VideoCapture(args.input)

    timestamps = []
    while True:
        b, img = cap.read() # slowest part is decoding the frame  
        if not b:
            break

        dtime, prob = detector.detect(img, verbose=args.verbose)
        timestamps.append(dtime)
        print(dtime)

    with open(args.out, 'w') as f: 
        st = [str(t)+'\n' for t in timestamps]
        f.writelines(st)


    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))