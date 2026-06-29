import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt

# load the dataset
DataClass = getattr(medmnist, INFO["organamnist"]["python_class"])
dataset = DataClass(split="train", download=True, size=64)

# grab one image and show it
img, label = dataset[0]
plt.imshow(img, cmap="gray")
plt.title("CT scan slice")
plt.show()