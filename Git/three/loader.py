import numpy as np
import torch
import os
import cv2
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import os
import cv2
import random
from PIL import Image
from torchvision import transforms



def augment(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(image)
    width, height = image.size

    # Apply random transformations
    if np.random.rand() > 0.5:
        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
    
    if np.random.rand() > 0.5:
        image = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))(image)

    if np.random.rand() > 0.5:
        image = transforms.RandomHorizontalFlip()(image)

    augmented_image = np.array(image)
    
    # Adjust points according to transformations
    augmented_points = points.copy()
    for i in range(4):
        x, y = points[2 * i], points[2 * i + 1]
        x_shift = np.random.uniform(-0.1, 0.1) * width
        y_shift = np.random.uniform(-0.1, 0.1) * height
        augmented_points[2 * i] += x_shift
        augmented_points[2 * i + 1] += y_shift
    
    return augmented_image, augmented_points


def resize(image: np.ndarray, points: np.ndarray, size: tuple[int, int] = (128, 128)) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(image)
    original_width, original_height = image.size
    resized_image = image.resize(size, Image.Resampling.LANCZOS)  # Updated this line
    
    scale_x = size[0] / original_width
    scale_y = size[1] / original_height
    
    resized_points = points.copy()
    for i in range(4):
        resized_points[2*i] *= scale_x
        resized_points[2*i+1] *= scale_y
    
    resized_image = np.array(resized_image)
    return resized_image, resized_points


def load(dir_name: str, image_size=(224, 224)) -> tuple[np.ndarray, np.ndarray]:
    images = []
    points = []

    image_dir = os.path.join(dir_name, 'images')
    label_dir = os.path.join(dir_name, 'labels')
    
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")

    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} does not exist.")
        return np.array(images), np.array(points)
    if not os.path.exists(label_dir):
        print(f"Label directory {label_dir} does not exist.")
        return np.array(images), np.array(points)
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)
            
            print(f"Processing image: {image_path}")
            print(f"Corresponding label file: {label_path}")
            
            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, image_size)
                    images.append(image)
                    
                    with open(label_path, 'r') as file:
                        pts = file.read().strip().split()
                        pts = [float(pt) for pt in pts]
                        
                        if len(pts) == 9:
                            print(f"Label file {label_filename} contains 9 values: {pts}")
                            # Extract the first 8 values
                            pts = pts[1:]
                        elif len(pts) != 8:
                            print(f"Skipping label file {label_filename}: contains {len(pts)} values instead of 8.")
                            continue
                        
                        points.append(pts)
                else:
                    print(f"Image {filename} could not be loaded.")
            else:
                print(f"Label file {label_filename} not found for image {filename}.")
    
    if len(images) == 0 or len(points) == 0:
        print("No images or points were loaded. Please check the dataset paths and ensure there are matching image-label pairs.")
    
    images = np.array(images)
    points = np.array(points)
    
    return images, points

# Test the load function
if __name__ == '__main__':
    data_dir = '/home/navid/Desktop/VisionProject/four-corners'
    images, points = load(data_dir)
    print(f"Loaded {len(images)} images and {len(points)} points.")


# Optional Data Generator
class PlateDataGenerator:
    def __init__(self, images, points, batch_size=32, augment=True, resize=True, size=(128, 128)):
        self.images = images
        self.points = points
        self.batch_size = batch_size
        self.augment = augment
        self.resize = resize
        self.size = size
        self.index = 0
    
    def __len__(self):
        return len(self.images) // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.images):
            self.index = 0
            raise StopIteration
        
        batch_images = []
        batch_points = []

        for _ in range(self.batch_size):
            image = self.images[self.index]
            points = self.points[self.index]

            self.index += 1

            if self.augment:
                image, points = augment(image, points)
            if self.resize:
                image, points = resize(image, points, self.size)
            batch_images.append(image)
            batch_points.append(points)
        
        batch_images = np.array(batch_images, dtype=np.float32) / 255.0
        batch_points = np.array(batch_points, dtype=np.float32)
        
        return torch.tensor(batch_images).permute(0, 3, 1, 2), torch.tensor(batch_points)
