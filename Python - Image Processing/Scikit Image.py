from skimage import io
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny
from matplotlib import pyplot as plt

img1 = io.imread("Image sample/inducedpluri.jpg")
img = io.imread("Image sample/inducedpluri.jpg", as_gray=True)
edge_roberts = roberts(img)
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)
edge_canny = canny(img)

#io.imsave("canny edge detection", edge_canny[0])

plt.figure(2)
plt.subplot(3,3,1)
plt.xlabel("Roberts Edge Detection")
plt.imshow(edge_roberts, cmap="gray")
plt.subplot(3,3,2)
plt.title("Edge Detection")
plt.xlabel("Canny Edge Detection")
plt.imshow(edge_canny, cmap="gray")
plt.subplot(3,3,3)
plt.xlabel("Sobel Edge Detection")
plt.imshow(edge_sobel, cmap="gray")
plt.subplot(2,2,3)
plt.xlabel("Scharr Edge Detection")
plt.imshow(edge_scharr, cmap="gray")
plt.subplot(2,2,4)
plt.xlabel("Prewitt Edge Detection")
plt.imshow(edge_prewitt, cmap="gray")

plt.figure(1)
plt.subplot(2,2,1)
plt.title("Medical Image Processing")
plt.xlabel("Real image")
plt.imshow(img1)
plt.subplot(2,2,2)
plt.xlabel("Grayscalled image")
plt.imshow(img, cmap="gray")

plt.show()
