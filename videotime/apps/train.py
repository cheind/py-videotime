

if __name__ == '__main__':
    import numpy as np
    import cv2
    import argparse
    import glob
    import os
    import videotime as vt

    def split(value):
        return [int(v) for v in value.split(',')]

    parser = argparse.ArgumentParser(description='Train time detector')
    parser.add_argument('indir', help='Directory containing images')
    parser.add_argument('--out', help='Name of resulting model file', default='videotime.npz')
    parser.add_argument('--digit-order', type=split, help='Digit order of images', default=np.arange(0,10,1,dtype=int))  
    parser.add_argument('--verbose', action='store_true')  
    args = parser.parse_args()
        
    files = glob.glob(os.path.join(args.indir, '*.png'))[:10]
    imgs = [cv2.imread(f) for f in files]

    model = vt.TimeDetector.train(imgs, digitorder=args.digit_order, verbose=args.verbose)
    print('Writing {}'.format(args.out))
    np.savez(args.out, **model)