import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from classifier import LicensePlateClassifier
import classifier
import numpy as np

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir
        self.labels = self.load_labels(label_file)
        self.image_files = list(self.labels.keys())
    def load_labels(self, label_file):
        labels = {}
        load(self.image_dir,label_file)

        # with open(label_file, 'r') as f:
        #     for line in f:
        #         parts = line.strip().split()
        #         filename = parts[0]
        #         label = parts[1]
        #         labels[filename] = [ord(char)-ord('A') if char.isalpha() else ord(char)-ord('0')+26 for char in label]
        # return labels
    
    
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 64)) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(self.labels[img_file], dtype=torch.long)
        return image, label

def train_classifier(image_dir, label_file, epochs=50, batch_size=32, model_path='classifier.pth'):
    dataset = LicensePlateDataset(image_dir, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = classifier.LicencePlateClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, 36), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def resize(image, points, new_size=(256, 256)):
    old_size = image.shape[:2]
    image = cv2.resize(image, new_size)
    points = points.reshape(-1, 2)
    points[:, 0] *= new_size[1] / old_size[1]
    points[:, 1] *= new_size[0] / old_size[0]

    image = image / 255.0

    points[:, 0] /= new_size[1]
    points[:, 1] /= new_size[0]

    return image, points.flatten()

def load(image_dir,label_file,num_variations=1):
    images = []
    points= []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_dir = os.path.join(image_dir,filename)
            label_dir = os.path.join(label_file,filename.replace('.jpg','.txt'))
            if os.path.exists(label_dir):
                image = cv2.imread(image_dir)
                if image is None:
                    continue
                try:
                    with open(label_dir,'r',coding = 'latin') as f:
                        yolo_data = f.read().strip().split()
                        if all(num.replace('.','',1).isdigit() for num in yolo_data):
                            yolo_data = [float(num) for num in yolo_data]
                        else: continue
                    if len(yolo_data) !=9:
                        continue
                    
                    img_height,img_width = image.shape[:2]
                    num_point = (len(yolo_data)-1)/2
                    point_array = np.zeros((num_point,2),dtype = np.float32)

                    for i in range(num_point):
                        x = yolo_data[2*i+1] * img_width
                        y = yolo_data[2*i+2] * img_height
                        point_array[i] = [x,y]

                    augmented_image,augmented_points = augment(image,point_array,num_variations)
                    for aug_image, aug_points in zip(augmented_image, augmented_points):
                        aug_image, aug_points = resize(aug_image, aug_points)
                        images.append(aug_image)
                        points.append(aug_points.flatten())
                except Exception as e:
                    print(e)
    return np.array(images), np.array(points)
#0..65
def augment(image, points, num_variations=1):
    h, w = image.shape[:2]
    points = points.reshape(-1, 2)
    augmented_images = []
    augmented_points = []

    for _ in range(num_variations):
        aug_image = image.copy()
        aug_points = points.copy()

        transformations = [
            {"type": "translation", "tx": np.random.uniform(-0.05 * w, 0.05 * w), "ty": np.random.uniform(-0.05 * h, 0.05 * h)},
            {"type": "blur", "ksize": np.random.choice([3, 5])}
        ]

        for transform in transformations:
            if transform["type"] == "translation":
                M = np.float32([[1, 0, transform["tx"]], [0, 1, transform["ty"]]])
                aug_image = cv2.warpAffine(aug_image, M, (w, h))
                aug_points += [transform["tx"], transform["ty"]]
            elif transform["type"] == "blur":
                aug_image = cv2.GaussianBlur(aug_image, (transform["ksize"], transform["ksize"]), 0)

        augmented_images.append(aug_image)
        augmented_points.append(aug_points.flatten())

    return augmented_images, augmented_points

if __name__ == '__main__':
    image_dir = '/home/navid/Desktop/VisionProject/four-corners/images'
    label_file = '/home/navid/Desktop/VisionProject/four-corners/labels'
    train_classifier(image_dir, label_file)
