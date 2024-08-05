import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from regressor import CNNModel
from loader import load, CustomDataset
import numpy as np

# Data Augmentation
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),  # Convert PIL Image back to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(data_dir, label_dir, model_path='model.pth', num_epochs=50, learning_rate=0.01, k=5):
    images, points = load(data_dir, label_dir)
    dataset = CustomDataset(images, points, transform=data_transforms)

    kfold = KFold(n_splits=k, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{k}')
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False)

        model = CNNModel()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, points in train_loader:
                images, points = images.to(device), points.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, points)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, points in val_loader:
                    images, points = images.to(device), points.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, points)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}')

        torch.save(model.state_dict(), f'{model_path}_fold{fold+1}.pth')
        print(f'Model for fold {fold+1} saved to {model_path}_fold{fold+1}.pth')

if __name__ == '__main__':
    data_dir = '/home/navid/Desktop/VisionProject/four-corners/images'
    label_dir = '/home/navid/Desktop/VisionProject/four-corners/labels'
    train_model(data_dir, label_dir)
