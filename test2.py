import numpy as np
import matplotlib as plt
from PIL import Image
import math

out_image = np.load('raw_out_image.npy')
q90, q10 = np.percentile(out_image, [99 ,1])
iqr = q90 - q10
out_image = np.where(out_image == 0., q10 - (iqr/4), out_image)
out_image = np.where(out_image > q90 + (iqr/4), q90 + (iqr/4), out_image)
out_image = np.where(out_image < q10 - (iqr/4), q10 - (iqr/4), out_image)
out_image[0,0,:] = q90 + (iqr/4)
out_image = out_image - np.min(out_image)
out_image = out_image * (255 / np.max(out_image))
Image.fromarray(out_image.astype(np.uint8)).save("out_image.bmp")