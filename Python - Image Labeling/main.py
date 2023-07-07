from skimage import io
import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny
from skimage import filters
import math

img1 = io.imread("eritrosit-normal1.jpg")
img = io.imread("eritrosit-normal1.jpg", as_gray='true')

def rgbtogray(parameter):
    return np.dot(parameter[...,:3], [0.2989, 0.5870, 0.1140])

def normal_value(parameter2):
    parameter2 = parameter2.astype(np.float64)/parameter2.max()
    parameter2 = 255*parameter2
    parameter2 = parameter2.astype(np.uint8)
    return parameter2

gray = rgbtogray(img1)
gray = normal_value(gray)

plt.figure(1)
plt.subplot(3,4,1)
plt.title("Real image")
plt.imshow(img1)
plt.axis('off')

plt.subplot(3,4,2)
plt.title("Grayscalled image")
plt.imshow(gray, cmap="gray")
plt.axis('off')

vertikal_kernel = np.array([[-1], [0], [1]])
gradient_vertikal = ndi.convolve(img, vertikal_kernel)
plt.subplot(3,4,3)
plt.title("Vertical Kernel")
plt.imshow(gradient_vertikal, cmap="gray")
plt.axis('off')

horizontal_kernel = vertikal_kernel.T
gradient_horizontal = ndi.convolve(img, horizontal_kernel)
plt.subplot(3,4,4)
plt.title("Horizontal Kernel")
plt.imshow(gradient_horizontal, cmap="gray")
plt.axis('off')

gradient_magnitude = np.sqrt(gradient_horizontal**2 + gradient_vertikal**2)
plt.subplot(3,4,5)
plt.title("Magnitude Kernel")
plt.imshow(gradient_magnitude, cmap="gray")
plt.axis('off')

edge_roberts = roberts(img)
plt.subplot(3,4,6)
plt.title("Roberts Filter")
plt.imshow(edge_roberts, cmap="gray")
plt.axis('off')

edge_sobel = sobel(img)
plt.subplot(3,4,7)
plt.title("Sobel Filter")
plt.imshow(edge_sobel, cmap="gray")
plt.axis('off')

edge_scharr = scharr(img)
plt.subplot(3,4,8)
plt.title("Scharr Filter")
plt.imshow(edge_scharr, cmap="gray")
plt.axis('off')

edge_prewitt = prewitt(img)
plt.subplot(3,4,9)
plt.title("Prewitt Filter")
plt.imshow(edge_prewitt, cmap="gray")
plt.axis('off')

edge_canny = canny(img)
plt.subplot(3,4,10)
plt.title("Canny Filter")
plt.imshow(edge_canny, cmap="gray")
plt.axis('off')

plt.figure(2)
plt.subplot(3,4,1)
plt.title("Real image")
plt.imshow(img1)
plt.axis('off')

plt.subplot(3,4,2)
plt.title("Grayscalled image")
plt.imshow(gray, cmap="gray")
plt.axis('off')

histogram = ndi.histogram(gray, min=0, max=255, bins=256)
plt.subplot(3,4,3)
plt.title("Histogram")
plt.plot(histogram)

threshold = filters.threshold_otsu(gray)
threshold = threshold*256
threshold1 = threshold = filters.threshold_otsu(img)
threshold1 = threshold1*256
#print(threshold)
print(threshold1)

plt.subplot(3,4,4)
plt.title("Noise Check 1")
plt.imshow(gray[100:130, 260:295], cmap='gray')
plt.colorbar()
plt.axis('off')
print('Standar Deviasi sebelum Median Filter', gray[100:130, 260:295].std())

median_filter = filters.median(gray, np.ones((7,7)))
plt.subplot(3,4,5)
plt.title("Median Filter")
plt.imshow(median_filter, cmap='gray')
plt.axis('off')

plt.subplot(3,4,6)
plt.title("Noise Check 2")
plt.imshow(median_filter[100:130, 260:295], cmap='gray')
plt.colorbar()
plt.axis('off')
print('Standar Deviasi sesudah Median Filter', median_filter[100:130, 260:295].std())

histogram2 = ndi.histogram(median_filter, min=0, max=255, bins=256)
plt.subplot(3,4,7)
#plt.title("Histogram 2")
plt.plot(histogram2)
plt.axis('off')

threshold2 = filters.threshold_otsu(median_filter)
plt.subplot(3,4,9)
plt.title("Contour dengan Median Filter")
plt.imshow(gray[:200, :200], cmap='gray')
plt.contour(median_filter[:200, :200], [threshold2])
print('Treshold setelah median Filter', threshold2)

plt.subplot(3,4,10)
plt.title("Contour dengan Subjetifitas")
plt.imshow(gray[:200, :200], cmap='gray')
plt.contour(gray[:200, :200], [170])

binary_image = gray < 170
plt.subplot(3,4,11)
plt.title("Binary Image dari Thresholding")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

from skimage import morphology

hanyabulatanbesar = morphology.remove_small_objects(binary_image, min_size=300)
hanyabulatanbesar2 = np.logical_not(morphology.remove_small_objects(np.logical_not(hanyabulatanbesar), min_size=300))
plt.subplot(3,4,12)
plt.title("Setelah menghilangkan yang kecil")
plt.imshow(hanyabulatanbesar2, cmap='gray')
plt.axis('off')

labels, nlabels = ndi.label(hanyabulatanbesar2)
print('Terdapat ', nlabels ,' Objek yang Terdeteksi')

fig, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (10,6))
for ii, obj_indices in enumerate(ndi.find_objects(labels)[15:21]):
    cell = hanyabulatanbesar2[obj_indices]
    axes[ii].imshow(cell, cmap='gray')
    axes[ii].axis('off')
    #axes[ii].title(ii+1)

plt.tight_layout()

from skimage.measure import label, regionprops_table, regionprops
import pandas as pd

label_image = label(hanyabulatanbesar2)
regions = regionprops(label_image)

fig, ax = plt.subplots()
ax.imshow(hanyabulatanbesar2, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=1)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=1)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=1)

ax.axis((0,575,461,0))

props = regionprops_table(label_image, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length'))
excel = pd.DataFrame(props)
print(props)

excel.to_excel("Exported_Dataframe.xlsx")

plt.show()