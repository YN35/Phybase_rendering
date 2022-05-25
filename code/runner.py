import numpy as np
from camera import Camera

if __name__=='__main__':
    np.set_printoptions(suppress=True)
    print(np.load('out_image.npy')[:,75,:])
    cm = Camera()
    out_image ,sdf_image = cm.render()
    final_image = cm.filter(out_image)
    cm.save_image(final_image,'out_image')
    cm.save_image(sdf_image,'sdf_image')