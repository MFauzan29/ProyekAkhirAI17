import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision.models import resnet18, ResNet18_Weights

# =============================
# 1. Konfigurasi Perangkat
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# =============================
# 2. Load dan Transform Data
# =============================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    image = (image_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.title(f"Label: {classes[label]}")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# =============================
# 4. Model ResNet (Tanpa ANN)
# =============================
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, 10)  # Ganti FC terakhir sesuai kelas CIFAR-10

    def forward(self, x):
        return self.model(x)

# =============================
# 5. Training Model
# =============================
def train_model():
    model = SimpleResNet().to(device)
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

            if (batch_idx + 1) % 100 == 0:
                print(f"[Epoch {epoch+1} | Batch {batch_idx+1}/{len(trainloader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(trainloader)
        print(f"Rata-rata Loss (Epoch {epoch+1}): {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'ResNet_model.pth')
            print("Loss menurun. Model disimpan.")
        else:
            epochs_no_improve += 1
            print(f"Tidak ada peningkatan signifikan ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("Early stopping dihentikan.")
            break

    print(f"\nTraining selesai. Loss terbaik yang dicapai: {best_loss:.4f}")
    return model

# =============================
# 6. Load atau Train
# =============================
def load_or_train_model():
    model = SimpleResNet().to(device)
    model_path = 'ResNet_model.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Memuat model yang sudah dilatih sebelumnya...")
    else:
        print("Model belum ada, melakukan training...")
        model = train_model()

    return model

# =============================
# 7. Evaluasi Model
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
# 8. Main Eksekusi
# =============================
if __name__ == "__main__":
    show_sample()
    trained_model = load_or_train_model()
    evaluate_model(trained_model)
