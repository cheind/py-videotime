import numpy as np
from datetime import datetime


class Layout:

    def __init__(self, **kwargs):
        self.digit_origin = np.asarray(kwargs.pop('digit_origin', [22, 6]))
        self.digit_shape = np.asarray(kwargs.pop('digit_shape', [21, 14]))
        self.digit_stride = np.asarray(kwargs.pop('digit_stride', [4+self.digit_shape[1], 0])) 
        self.digit_ids = kwargs.pop('digit_ids', [0,1,3,4,6,7,9,10])
        self.ndigits = len(self.digit_ids)

        # Image slices for each digit location
        self.slicesx = []
        self.slicesy = []
        for i in self.digit_ids:
            xy = self.digit_origin + i*self.digit_stride            
            self.slicesx.append(slice(xy[0], xy[0] + self.digit_shape[1]))
            self.slicesy.append(slice(xy[1], xy[1] + self.digit_shape[0]))

    def build_time(self, dtime):
        d = datetime.now()
        h = 10*dtime[0] + dtime[1]
        m = 10*dtime[2] + dtime[3]
        s = 10*dtime[4] + dtime[5]
        micro = (10*dtime[6] + dtime[7])*int(1e4)
        return d.replace(hour=h, minute=m, second=s, microsecond=micro)        
        
    
if __name__ == '__main__':
    import numpy as np
    import cv2
    import argparse
    import glob
    import os

    
    parser = argparse.ArgumentParser(description='Train time detector')
    parser.add_argument('indir', help='Directory containing images')    
    args = parser.parse_args()

    f = glob.glob(os.path.join(args.indir, '*.png'))[0]
    img = cv2.imread(f)

    l = Layout()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, l.ndigits)
    for i in range(l.ndigits):
        axs[i].imshow(img[l.slicesy[i], l.slicesx[i],0], origin='upper')
        axs[i].axis('off')
    plt.show()


    