import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np

DataClass = getattr(medmnist, INFO["organamnist"]["python_class"])
dataset = DataClass(split="train", download=True, size=64)

img, label = dataset[0]
original = np.asarray(img) / 255.0  # divide so values are 0-1

# create a grid of noise the same shape as the image
noise = np.random.uniform(-0.05, 0.05, size=original.shape)

# add it
corrupted = original + noise


# motion blur. averageing each pixel with its neighbhours makes the bright spots more evened out
from scipy.ndimage import uniform_filter

blurred = uniform_filter(original, size=5)


#missing data

mask = (np.random.rand(*original.shape) > 0.3) #with np.random.rand having a parameter it sets a grid or random number with size og
missing = original * mask
# np.random.rand() > 0.3 is false 30% of the time. false is 0 so 0 times any number is 0. this will happen 30% of the time

#displaying the images
fig, axes = plt.subplots(1, 4)
axes[0].imshow(original, cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(corrupted, cmap="gray")
axes[1].set_title("Noisy")
axes[2].imshow(blurred, cmap="gray")
axes[2].set_title("Blurred")
axes[3].imshow(missing, cmap="gray")
axes[3].set_title("Missing Pixels")
plt.show()

