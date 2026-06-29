import torch
import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
from model import DenoisingAutoencoder

# load the saved model
model = DenoisingAutoencoder()
model.load_state_dict(torch.load("model_weights.pt"))
model.eval()

# load a few test images (these are new images the model never saw)
DataClass = getattr(medmnist, INFO["organamnist"]["python_class"])
test_data = DataClass(split="test", download=True, size=64)

def corrupt(img):
    noise = np.random.uniform(-0.1, 0.1, size=img.shape)
    return np.clip(img + noise, 0, 1)

# show 4 examples
fig, axes = plt.subplots(4, 3, figsize=(8, 12))
axes[0, 0].set_title("Corrupted")
axes[0, 1].set_title("Reconstructed")
axes[0, 2].set_title("Original")

for i in range(4):
    img, _ = test_data[i]
    original = np.asarray(img, dtype=np.float32) / 255.0
    corrupted = corrupt(original)

    input_tensor = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    
    reconstructed = output.squeeze().numpy()

    axes[i, 0].imshow(corrupted, cmap="gray")
    axes[i, 1].imshow(reconstructed, cmap="gray")
    axes[i, 2].imshow(original, cmap="gray")
    for j in range(3):
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()

# stuff after this for getting an immage

from PIL import Image
import medmnist
from medmnist import INFO
import numpy as np

DataClass = getattr(medmnist, INFO["organamnist"]["python_class"])
test_data = DataClass(split="test", download=True, size=64)

img, _ = test_data[0]
img.save("sample_ct.png")
print("Saved sample_ct.png")