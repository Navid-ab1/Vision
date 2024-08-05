import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import random
# def test (image_path = '/home/navid/Desktop/VisionProject/Test_image'):
#     image = cv2.imread(image_path)
#     points = np.array([[0.33375111849641503, 0.44584873554048715],[0.7320580100203828 , 0.442254517965083],[
#         0.7308046545311376, 0.5506102212267183],[0.3347763347763347 ,0.5519230769230771]])
#     augment(image,points)
#     augment_image = []
def augment(image, points, num_variations=1,save_dir=None,image_name=None):
    h, w = image.shape[:2]
    points = points.reshape(-1, 2)
    augmented_images = []
    augmented_points = []

    for i in range(num_variations):
        aug_image = image.copy()
        aug_points = points.copy()

        transformations = [
            {"type": "translation", "tx": np.random.uniform(-0.05 * w, 0.05 * w), "ty": np.random.uniform(-0.05 * h, 0.05 * h)},
            {"type": "blur", "ksize": np.random.choice([3, 5])},
            {"type": "rotation", "angle": np.random.uniform(-15, 15)},
            {"type": "contrast", "alpha": np.random.uniform(0.8, 1.2)},
            {"type": "crop", "crop_size": (random.randint(224, 256), random.randint(224, 256))},
            {"type": "resize", "size": (256, 256)},
            {"type": "perspective", "points": points * np.random.uniform(0.95, 1.05, points.shape)}
        ]

        for transform in transformations:
            if transform["type"] == "translation":
                M = np.float32([[1, 0, transform["tx"]], [0, 1, transform["ty"]]])
                aug_image = cv2.warpAffine(aug_image, M, (w, h))
                aug_points += [transform["tx"], transform["ty"]]
            elif transform["type"] == "blur":
                aug_image = cv2.GaussianBlur(aug_image, (transform["ksize"], transform["ksize"]), 0)
            elif transform["type"] == "rotation":
                M = cv2.getRotationMatrix2D((w / 2, h / 2), transform["angle"], 1)
                aug_image = cv2.warpAffine(aug_image, M, (w, h))
                cos_angle = np.cos(np.deg2rad(transform["angle"]))
                sin_angle = np.sin(np.deg2rad(transform["angle"]))
                aug_points = np.dot(aug_points - np.array([w / 2, h / 2]), np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])) + np.array([w / 2, h / 2])
            elif transform["type"] == "contrast":
                aug_image = cv2.convertScaleAbs(aug_image, alpha=transform["alpha"], beta=0)
            elif transform["type"] == "crop":
                crop_h, crop_w = transform["crop_size"]
                start_x = np.random.randint(0, w - crop_w)
                start_y = np.random.randint(0, h - crop_h)
                aug_image = aug_image[start_y:start_y + crop_h, start_x:start_x + crop_w]
                aug_points = (aug_points - np.array([start_x, start_y])) * np.array([w / crop_w, h / crop_h])
            elif transform["type"] == "resize":
                aug_image = cv2.resize(aug_image, transform["size"])
                aug_points = aug_points * np.array([transform["size"][1] / w, transform["size"][0] / h])
            elif transform["type"] == "perspective":
                pts1 = np.float32(points)
                pts2 = np.float32(transform["points"])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                aug_image = cv2.warpPerspective(aug_image, M, (w, h))
                aug_points = cv2.perspectiveTransform(aug_points[None, :, :], M)[0]
        if save_dir and image_name:
            save_aug(aug_image,save_dir,image_name,i)
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

                    # # augmented_images, augmented_points = augment(image, point_array, num_variations,save_dir,filename)
                    # for aug_image,aug_points in zip(aug_image,augmented_points):
                    #     aug_image,aug_points = resize(aug_image,aug_points)
                    #     images.append(aug_image)
                    #     points.append(aug_points.flatten())

                    # for aug_image, aug_points in zip(augmented_images, augmented_points):
                    #     aug_image, aug_points = resize(aug_image, aug_points)
                    #     images.append(aug_image)
                    #     points.append(aug_points.flatten())
                except Exception as e:
                    print(f"Error processing file {label_path}: {str(e)}")

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
def save_aug(image,save_dir,or_name,idx):
    if not os.path.exists('/home/navid/Desktop/VisionProject/Quera/augment_image'):
        os.makedirs(save_dir)
    base_name = os.path.splitext(or_name)[0]
    aug_ima = f"{base_name}_aug_{idx}.jpg"
    save_path = os.path.join(save_dir,aug_ima)
    cv2.imwrite(save_path,image)
