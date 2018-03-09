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

        # Extract letters from HH:MM:SS.TT
        rois = np.array([img[ys, xs] for xs, ys in zip(self.xslices, self.yslices)])

        # Scale/shift to [-1, 1]
        srois = (rois / 255. - 0.5)*2
        
        # Inner product of templates with letter positions to yield scores
        scores = np.tensordot(srois, self.letters, axes=([1,2], [1,2]))

        # Probabilities for each roi according to the alphabet (softmax)
        def softmax(x):
            # e ^ (x - max(x)) / sum(e^(x - max(x))
            xn = x - x.max(axis=1, keepdims=True)
            ex = np.exp(xn)
            return  ex / ex.sum(axis=1, keepdims=True)

        probs = softmax(scores)
        print(probs.max(axis=1))

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
        digitorder = kwargs.pop('digitorder', list(range(10)))
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
    parser.add_argument('--noise', type=float, help='noise level')
    parser.add_argument('--shiftx', type=int, help='shift image in x')
    parser.add_argument('--shifty', type=int, help='shift image in y')
    args = parser.parse_args()

    model = np.load(args.model)
    detector = TimeDetector(layout=model['layout'].item(), letters=model['letters'])

    files = glob.glob(os.path.join(args.indir, '*.png'))
    for f in files:
        img = cv2.imread(f)
        if args.noise:
            img = img.astype(float) / 255 - 0.5
            img += np.random.normal(0, scale=args.noise, size=img.shape)
            img = np.clip((img+0.5)*255, 0, 255).astype(np.uint8)
        if args.shiftx:
            img = np.roll(img, np.random.randint(-args.shiftx, args.shiftx), axis=1)
        if args.shifty:
            img = np.roll(img, np.random.randint(-args.shifty, args.shifty), axis=0)
        
        detector.detect(img, verbose=args.verbose)