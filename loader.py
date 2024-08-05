# loader.py
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

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

def load(image_dir, label_dir, num_variations=1):
    images = []
    points = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                try:
                    with open(label_path, 'r', encoding='latin1') as f:
                        yolo_data = f.read().strip().split()
                        if all(num.replace('.', '', 1).isdigit() for num in yolo_data):
                            yolo_data = [float(num) for num in yolo_data]
                        else:
                            continue
                    if len(yolo_data) != 9:
                        continue

                    img_height, img_width = image.shape[:2]
                    num_points = (len(yolo_data) - 1) // 2
                    point_array = np.zeros((num_points, 2), dtype=np.float32)

                    for i in range(num_points):
                        x = yolo_data[1 + 2*i] * img_width
                        y = yolo_data[1 + 2*i + 1] * img_height
                        point_array[i] = [x, y]

                    augmented_images, augmented_points = augment(image, point_array, num_variations)

                    for aug_image, aug_points in zip(augmented_images, augmented_points):
                        aug_image, aug_points = resize(aug_image, aug_points)
                        images.append(aug_image)
                        points.append(aug_points.flatten())
                except Exception as e:
                    print(f"Error processing file {label_path}: {str(e)}")

    return np.array(images), np.array(points)

class CustomDataset(Dataset):
    def __init__(self, images, points, transform=None):
        self.images = images
        self.points = points
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        points = self.points[idx]

        if self.transform:
            image = self.transform(image)

        points = torch.tensor(points, dtype=torch.float32)

        return image, points
