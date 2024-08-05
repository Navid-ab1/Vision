import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from regressor import CNNModel
from loader import load, CustomDataset

def train_model(data_dir, label_dir, epochs=50, batch_size=32, model_path='model.pth'):
    images, points = load(data_dir, label_dir)
    dataset = CustomDataset(images, points)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, points in val_loader:
                images, points = images.to(device), points.to(device)
                outputs = model(images)
                loss = criterion(outputs, points)
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    data_dir = '/home/navid/Desktop/VisionProject/four-corners/images'
    label_dir = '/home/navid/Desktop/VisionProject/four-corners/labels'
    train_model(data_dir, label_dir)
