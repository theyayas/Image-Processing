from skimage import io
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np


img1 = io.imread("Citra1.jpg")
img = io.imread("Citra1.jpg", as_gray = True)

print(np.shape(img1))
print(type(img1))

plt.figure(1)
plt.subplot(2,2,1)
plt.title("Medical Image Processing")
plt.xlabel("Real image")
plt.imshow(img1)
plt.subplot(2,2,2)
plt.title("Medical Image Processing")
plt.xlabel("Real image")
plt.imshow(img, cmap = 'gray')

plt.show()
