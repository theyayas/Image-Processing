import numpy as np
from matplotlib import pyplot as plt

random_image = np.random.random([500,500])

plt.imshow(random_image, cmap='gray')
plt.colorbar()
plt.show()