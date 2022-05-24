import numpy as np
from camera import Camera

if __name__=='__main__':
    np.set_printoptions(suppress=True)
    print(np.load('out_image.npy')[:,37,:])
    cm = Camera()
    cm.render()
    