import numpy as np
import pickle
from videotime.layout import Layout

class TimeDetector:
    def __init__(self, layout, weights):
        self.layout = layout
        self.weights = weights

    def detect(self, img, **kwargs):
        verbose = kwargs.pop('verbose', False)

        img = np.asarray(img)
        if img.ndim == 3:
            img = img[..., 0]        

        # Extract letters from HH:MM:SS.TT
        rois = np.array([img[ys, xs] for xs, ys in zip(self.layout.slicesx, self.layout.slicesy)])

        # Scale/shift to [-1, 1]
        srois = (rois / 255. - 0.5)*2
        
        # Inner product of digit positions and weights to yield scores
        scores = np.tensordot(srois, self.weights, axes=([1,2], [1,2]))

        # Probabilities for each roi according to the alphabet (softmax)
        def softmax(x):
            # e ^ (x - max(x)) / sum(e^(x - max(x))
            xn = x - x.max(axis=1, keepdims=True)
            ex = np.exp(xn)
            return  ex / ex.sum(axis=1, keepdims=True)

        # Use max probs for each digit position as detection result
        probs = softmax(scores)
        dtime = np.argmax(probs, axis=1)       
        dprobs = probs[np.arange(probs.shape[0]), dtime]

        try:
            dtime = self.layout.build_time(dtime)
        except ValueError:
            dtime = None

        if verbose and dtime is not None:
            from datetime import datetime
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, rois.shape[0])
            fig.suptitle('Detected time {} - {}%'.format(datetime.strftime(dtime, '%H:%M:%S.%f'), dprobs.min()*100))
            [ax.axis('off') for ax in axs]
            [ax.imshow(l, origin='upper') for ax,l in zip(axs, rois)]
            plt.show()

        return dtime, dprobs.min()

    def save(self, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def train(imgs, **kwargs):
        digitorder = kwargs.pop('digitorder', list(range(10)))
        layout = kwargs.pop('layout', Layout())
        secondid = kwargs.pop('secondid', 7)        
        verbose = kwargs.pop('verbose', False)
        xshifts = kwargs.pop('xshifts', 0)
        yshifts = kwargs.pop('yshifts', 0)

        so = np.argsort(digitorder)
        imgs = np.asarray(imgs)[so]
        
        if imgs.ndim == 4:
            imgs = imgs[..., 0]

        imgs = (imgs / 255. - 0.5)*2

        idx = layout.digit_ids.index(secondid)
        
        rois = []
        for xs in range(-xshifts,xshifts+1):
            for ys in range(-yshifts,yshifts+1):
                rimgs = np.roll(imgs, xs, axis=2)
                rimgs = np.roll(rimgs, ys, axis=1)
                rois.append(rimgs[:, layout.slicesy[idx], layout.slicesx[idx]])
        nshifts = (2*xshifts+1) * (2*yshifts+1)
        rois = np.concatenate(rois).reshape(nshifts, 10, layout.digit_shape[0], layout.digit_shape[1])
        rois = np.mean(rois, axis=0)

        weights = rois

        if verbose:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 10)
            [ax.axis('off') for ax in axs]
            [ax.imshow(r, origin='upper') for ax,r in zip(axs, rois)]
            plt.show()

        return TimeDetector(layout, weights)


if __name__ == '__main__':
    import numpy as np
    import cv2
    import argparse
    import glob
    import os

    def split(value):
        return [int(v) for v in value.split(',')]

    parser = argparse.ArgumentParser(description='Train time detector')
    parser.add_argument('model', help='Model file')
    parser.add_argument('indir', help='Directory containing images')    
    parser.add_argument('--verbose', action='store_true')  
    parser.add_argument('--noise', type=float, help='noise level')
    parser.add_argument('--shiftx', type=int, help='shift image in x')
    parser.add_argument('--shifty', type=int, help='shift image in y')
    args = parser.parse_args()

    detector = TimeDetector.load(args.model)
    
    files = glob.glob(os.path.join(args.indir, '*.png'))
    for f in files:
        img = cv2.imread(f)
        if args.noise:
            img = img.astype(float) / 255 - 0.5
            img += np.random.normal(0, scale=args.noise, size=img.shape)
            img = np.clip((img+0.5)*255, 0, 255).astype(np.uint8)
        if args.shiftx:
            img = np.roll(img, args.shiftx, axis=1)
        if args.shifty:
            img = np.roll(img, args.shifty, axis=0)
        
        detector.detect(img, verbose=args.verbose)