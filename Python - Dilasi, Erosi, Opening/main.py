from skimage import io, exposure
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

#memuat gambar
image = io.imread('Citra4.jpg')

def rgbtogray(parameter):
    return np.dot(parameter[...,:3], [0.2989, 0.5870, 0.1140])

def normal_value(parameter2):
    parameter2 = parameter2.astype(np.float64)/parameter2.max()
    parameter2 = 255*parameter2
    parameter2 = parameter2.astype(np.uint8)
    return parameter2

gray = rgbtogray(image)
gray = normal_value(gray)

print('Data Type : ', gray.dtype)
print('Min. Value : ', gray.min())
print('Max. Value : ', gray.max())
print(gray.shape)

#membuat histogram gambar
"""
histogram = ndi.histogram(gray, min = 0, max = 255, bins = 256)
plt.subplot(2,2,3)
plt.title("Real Histogram")
plt.plot(histogram)

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.title("Image")
plt.axis('off')"""

#AHE
image_equalization = exposure.equalize_adapthist(image, clip_limit=0.1)
image_equalization = normal_value(image_equalization)
plt.subplot(2,3,1)
plt.title("Equalized Image")
plt.imshow(image_equalization)
plt.axis('off')

histogram2 = ndi.histogram(image_equalization, min=0, max=255, bins=256)
plt.subplot(2,3,4)
plt.title("Equalized Histogram")
plt.plot(histogram2)

#contras streching
"""
a = 0
b = 255
c = 8
d = 150

streched_image = (image - c)*((b-a)/(d-c)) + a
plt.subplot(2,3,3)
plt.title("Streched Contras Image")
plt.imshow(streched_image)

histogram3 = ndi.histogram(streched_image, min=0, max=255, bins=256)
plt.subplot(2,3,6)
plt.title("Streched Contras Image Histogram")
plt.plot(histogram3) """

#masking
#mask_tulang = (gray >= 6) & (gray <= 8)
mask_tulang = np.where(gray >= 145, 1, 0)
dilatasi_tulang = ndi.binary_dilation(mask_tulang, iterations=5)
close_tulang = ndi.binary_closing(mask_tulang, iterations=5)
erosi_tulang = ndi.binary_erosion(mask_tulang, iterations=5)
open_tulang = ndi.binary_opening(mask_tulang, iterations=5)

#plt.figure(2)
plt.subplot(2,3,2)
plt.title("Dilatasi")
plt.imshow(dilatasi_tulang, cmap='gray')
plt.axis('off')
plt.subplot(2,3,3)
plt.title("Close")
plt.imshow(close_tulang, cmap='gray')
plt.axis('off')
plt.subplot(2,3,5)
plt.title("Erosi")
plt.imshow(erosi_tulang, cmap='gray')
plt.axis('off')
plt.subplot(2,3,6)
plt.title("Open")
plt.imshow(open_tulang, cmap='gray')
plt.axis('off')

plt.show()