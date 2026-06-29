import torch
import torch.nn as nn
import numpy as np
import medmnist
from medmnist import INFO
from model import DenoisingAutoencoder
from scipy.ndimage import uniform_filter

# load dataset
DataClass = getattr(medmnist, INFO["organamnist"]["python_class"])
train_data = DataClass(split="train", download=True, size=64)
val_data = DataClass(split="val", download=True, size=64)

from scipy.ndimage import uniform_filter

def corrupt(img):
    choice = np.random.randint(0, 3)
    if choice == 0:
        # noise
        noise = np.random.uniform(-0.1, 0.1, size=img.shape)
        return np.clip(img + noise, 0, 1)
    elif choice == 1:
        # blur
        return uniform_filter(img, size=5)
    else:
        # missing pixels
        mask = (np.random.rand(*img.shape) > 0.3)
        return img * mask

def prepare(dataset):
    pairs = []
    for img, _ in dataset:
        clean = np.asarray(img, dtype=np.float32) / 255.0
        noisy = corrupt(clean)
        # convert to tensors with shape (1, 64, 64)
        clean_t = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        noisy_t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        pairs.append((noisy_t, clean_t))
    return pairs

print("Preparing data...")
train_pairs = prepare(train_data)
val_pairs = prepare(val_data)
print(f"Training images: {len(train_pairs)}, Validation: {len(val_pairs)}")

model = DenoisingAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training...")
for epoch in range(30):
    model.train()
    total_loss = 0
    for noisy, clean in train_pairs:
        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)
        output = model(noisy)
        loss = loss_fn(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_pairs)
    print(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.5f}")

torch.save(model.state_dict(), "model_weights.pt")
print("Done! Saved model_weights.pt")