import numpy as np
import pickle
from videotime.layout import Layout

class TimeDetector:
    '''Detects superimposed timestamps in videos.
    
    The detector performs a trivial template match to determine
    the digit class for each digit position in a given Layout.
    
    Since overlay digits are usually rendered pixel exact, the
    detector considers only a single image location per digit.

    Extracting the time of a single still image usually takes around 
    80usecs, or equivalently 12500 images can be processed per second.
    The detector is quite robust against Gaussian noise but cannot handle
    images shifts very well. However, this usually does not pose a problem
    as overlays are rendered exactly.
    '''

    def __init__(self, layout, weights):
        '''Create detector from time layout and weights.'''
        self.layout = layout
        self.weights = weights

    def detect(self, img, **kwargs):
        '''Detect and parse time overlay.

        Params
        ------
        img : HxW, HxWxC
            Image to be processed
        
        Kwargs
        ------
        verbose: bool, optional
            Wheter or not to sho detection result.

        Returns
        -------
        time : datetime object
            Detected time
        probability: scalar
            Probability of detection [0..1]
        '''

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
        '''Save detector'''
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(fname):
        '''Load detector'''
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def train(imgs, **kwargs):
        '''Train detector from images.

        This function builds a model for each digit (0-9). It assumes
        10 images to be handed. A layout is assumed that describes the 
        positioning of the time overlay in images. Each image is assumed
        to contain a single unique digit in a specific position according
        to the layout. The position can be controlled via the `digitid` 
        parameter. If the order of the images is not in digit-order, 
        `digitorder` can be used to correct the order.

        Params
        ------
        imgs : 10xHxW, 10xHxWxC
            10 images containing all digits of 0-9. Each image
            contributes one unique digit. The digit location is controlled
            via `digitid`.
        
        Kwargs
        ------
        digitorder: array-like, optional
            Defines which image contains which digit. By default it is 
            assumed that the images contain digits in order. So first
            image contributes 0 digit, second image provides digit 1, etc.
        verbose: bool, optional
            Whether or not to show extracted digits for debugging.
        xshifts: int
            Specifies the amount of +/- shifting in x-direction of images,
            to compensate for potential image shifts. Usually not necessary 
            as overlays are rendered at exact positions.
        yshifts: int
            Specifies the amount of +/- shifting of images in y-direction
            to compensate for potential image shifts. Usually not necessary 
            as overlays are rendered at exact positions.

        Returns
        -------
        detector : TimeDetector object
            The trained detector.
        '''
        
        digitorder = kwargs.pop('digitorder', list(range(10)))
        layout = kwargs.pop('layout', Layout())
        digitid = kwargs.pop('digitid', 7) # For Axis cameras this is the minor second digit.
        verbose = kwargs.pop('verbose', False)
        xshifts = kwargs.pop('xshifts', 0)
        yshifts = kwargs.pop('yshifts', 0)

        so = np.argsort(digitorder)
        imgs = np.asarray(imgs)[so]
        
        if imgs.ndim == 4:
            imgs = imgs[..., 0]

        imgs = (imgs / 255. - 0.5)*2

        idx = layout.digits.index(digitid)
        
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
            # You should see digits 0..9 in order.

        return TimeDetector(layout, weights)