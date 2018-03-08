import numpy as np

from videotime.common import Alphabet

class TimeDetector:
    def __init__(self, layout, letters, **kwargs):
        self.layout = layout
        self.letters = letters
        self.xslices = [self.xyslice(self.layout, i)[0] for i in range(self.layout['npos'])]
        self.yslices = [self.xyslice(self.layout, i)[1] for i in range(self.layout['npos'])]

    def detect(self, img, **kwargs):
        verbose = kwargs.pop('verbose', False)

        if img.ndim == 3:
            img = img[..., 0]
        
        img = (img / 255. - 0.5)*2

        # Extract letters from HH:MM:SS.TT
        rois = np.array([img[ys, xs] for xs, ys in zip(self.xslices, self.yslices)])

        scores = np.einsum('ljk, ijk ->li', rois, self.letters)

        def softmax(x):
            return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

        # Probabilities for each roi according to the alphabet
        probs = softmax(scores)

        # Use max probs for each position as detection result
        dtime = np.argmax(probs, axis=1)       
        dprobs = probs[np.arange(probs.shape[0]), dtime]
 
        strtime = '{}{}:{}{}:{}{}.{}{}'.format(
            dtime[0],dtime[1],dtime[3], 
            dtime[4],dtime[6],dtime[7],
            dtime[9],dtime[10])
        
        if verbose:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, rois.shape[0])
            fig.suptitle('Detected time {} - {}%'.format(strtime, dprobs.min()))
            [ax.axis('off') for ax in axs]
            [ax.imshow(l, origin='upper') for ax,l in zip(axs, rois)]
            plt.show()

        return dtime, dprobs.min(), strtime
        

    @staticmethod
    def train(imgs, **kwargs):
        digitorder = kwargs.pop('digitorder', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        layout = kwargs.pop('layout', {'x':22, 'y':6, 'w':14, 'h':21, 'rpad':4, 'npos' : 11})
        verbose = kwargs.pop('verbose', False)
        
        # Order input images
        so = np.argsort(digitorder)
        imgs = np.array(imgs, dtype=np.float)[so]
        imgs = (imgs / 255. - 0.5)*2

        rois = []
        xslices = []
        yslices = []

        # Extract second digits from last second pos HH:MM:SS.TT
        xs, ys = TimeDetector.xyslice(layout, 7)        
        rois.append(imgs[:, ys, xs, 0])
        xslices.append(xs)
        yslices.append(ys)

        # :
        xs, ys = TimeDetector.xyslice(layout, 5)
        rois.append(np.expand_dims(imgs[0, ys, xs, 0], 0))
        xslices.append(xs)
        yslices.append(ys)

        # .
        xs, ys = TimeDetector.xyslice(layout, 8)
        rois.append(np.expand_dims(imgs[0, ys, xs, 0], 0))
        xslices.append(xs)
        yslices.append(ys)

        letters = np.concatenate(rois) # 12xHxW

        if verbose:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, layout['npos'])
            [ax.axis('off') for ax in axs]
            [ax.imshow(l, origin='upper') for ax,l in zip(axs, letters)]
            plt.show()

        return {'layout' : layout, 'letters' : letters}
        
    @staticmethod
    def xyslice(layout, idx): 
        '''Extract the x/y slice for i-th letter in the image.'''       
        xs = layout['x'] + (layout['w'] + layout['rpad'])*idx
        return slice(xs, xs+layout['w']), slice(layout['y'], layout['y'] + layout['h'])

    @staticmethod
    def load(fname):
        model = np.load(fname)
        return TimeDetector(layout=model['layout'].item(), letters=model['letters'])

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
    args = parser.parse_args()

    model = np.load(args.model)
    detector = TimeDetector(layout=model['layout'].item(), letters=model['letters'])

    files = glob.glob(os.path.join(args.indir, '*.png'))
    for f in files:
        detector.detect(cv2.imread(f), verbose=args.verbose)