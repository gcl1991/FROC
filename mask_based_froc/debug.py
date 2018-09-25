import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

for i in range(3):
    img = mpimg.imread('ground_truth.png')[:,:,i]
    print img.max()

