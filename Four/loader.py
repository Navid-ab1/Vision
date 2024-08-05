import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def augment(image, points, num_variations=1):
    h, w = image.shape[:2]
    points = points.reshape(-1, 4)
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
                aug_points[:, [0, 1]] += [transform["tx"] / w, transform["ty"] / h]
            elif transform["type"] == "blur":
                aug_image = cv2.GaussianBlur(aug_image, (transform["ksize"], transform["ksize"]), 0)

        augmented_images.append(aug_image)
        augmented_points.append(aug_points.flatten())

    return augmented_images, augmented_points

def resize(image, points, new_size=(256, 256)):
    old_size = image.shape[:2]
    image = cv2.resize(image, new_size)
    points = points.reshape(-1, 4)
    points[:, [0, 2]] *= new_size[1] / old_size[1]
    points[:, [1, 3]] *= new_size[0] / old_size[0]

    image = image / 255.0

    points[:, [0, 2]] /= new_size[1]
    points[:, [1, 3]] /= new_size[0]

    return image, points.flatten()

def parse_xml(xml_path, img_width, img_height):
    with open(xml_path, 'r', encoding='latin1') as f:
        objects = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                xmin = float(parts[1]) * img_width
                ymin = float(parts[2]) * img_height
                xmax = xmin + float(parts[3]) * img_width
                ymax = ymin + float(parts[4]) * img_height
                objects.append((class_id, xmin, ymin, xmax, ymax))
        return objects

def load(image_dir, xml_dir, num_variations=1):
    images = []
    points = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            xml_path = os.path.join(xml_dir, filename.replace(".jpg", ".xml"))

            if os.path.exists(xml_path):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                img_height, img_width = image.shape[:2]
                objects = parse_xml(xml_path, img_width, img_height)

                points_array = np.array([
                    [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
                    for _, xmin, ymin, xmax, ymax in objects
                ])

                augmented_images, augmented_points = augment(image, points_array, num_variations)

                for aug_image, aug_points in zip(augmented_images, augmented_points):
                    aug_image, aug_points = resize(aug_image, aug_points)
                    images.append(aug_image)
                    points.append(aug_points.flatten())
    
    return np.array(images), np.array(points)

class CustomDataset(Dataset):
    def __init__(self, images, points):
        self.images = images
        self.points = points

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        points = self.points[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        points = torch.tensor(points, dtype=torch.float32)

        return image, points
