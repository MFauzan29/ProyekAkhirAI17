import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =============================
# 1. Konfigurasi Perangkat
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())  # Harusnya keluar: True
print(torch.cuda.get_device_name(0))  # Nama GPU lo (misal: NVIDIA GeForce GTX 1650)

# =============================
# 2. Load dan Transform Data
# =============================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisasi RGB
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
           5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# =============================
# 3. Tampilkan Gambar Sample
# =============================
def show_sample():
    image_tensor, label = train_data[0]
    image = (image_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()  # unnormalize
    plt.imshow(image)
    plt.title(f"Label: {classes[label]}")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# =============================
# 4. Definisi ANN
# =============================
from torchvision.models import resnet18, ResNet18_Weights

class ResNetANN(nn.Module):
    def __init__(self):
        super(ResNetANN, self).__init__()
        
        resnet = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Tanpa layer FC terakhir
        
        # ANN custom (input 512 karena output dari ResNet terakhir adalah 512-dim)
        self.fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.feature_extractor(x)  # Output shape: [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)      # Flatten ke [batch_size, 512]
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x


# =============================
# 5. Training Model
# =============================
def train_model():
    model = ResNetANN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    epochs = 50
    patience = 5
    min_delta = 0.001
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ✅ Print loss setiap 100 batch
            if (batch_idx + 1) % 100 == 0:
                print(f"[Epoch {epoch+1} | Batch {batch_idx+1}/{len(trainloader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(trainloader)

        # ✅ Print rata-rata loss per epoch
        print(f"Rata-rata Loss (Epoch {epoch+1}): {avg_loss:.4f}")

        # Early stopping
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'ResNetANN_model.pth')
            print("Loss menurun. Model disimpan.")
        else:
            epochs_no_improve += 1
            print(f"Tidak ada peningkatan signifikan ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("Early stopping dihentikan.")
            break

    # ✅ Log akhir setelah training
    print(f"\nTraining selesai. Loss terbaik yang dicapai: {best_loss:.4f}")

    return model

import os

def load_or_train_model():
    model = ResNetANN().to(device)
    model_path = 'ResNetANN_model.pth'

    if os.path.exists(model_path):
        print("Memuat model yang sudah dilatih sebelumnya...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Model belum ada, melakukan training...")
        model = train_model()

    return model

# =============================
# 6. Evaluasi Model
# =============================
def evaluate_model(model):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nAkurasi di data test: {accuracy:.2f}%")

# =============================
# 7. Main Eksekusi
# =============================
if __name__ == "__main__":
    show_sample()
    trained_model = load_or_train_model()
    evaluate_model(trained_model)
