import os
import torch
import numpy as np
import cv2
from regressor import CNNModel
from classifier import LicensePlateClassifier
from loader import load, CustomDataset
from torch.utils.data import DataLoader

# Function to load Persian license plate characters
def load_persian_characters():
    persian_chars = '۱۲۳۴۵۶۷۸۹۰' + 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
    return persian_chars

# Function to map labels to Persian characters
def map_to_persian_chars(indices):
    persian_chars = load_persian_characters()
    return ''.join(persian_chars[idx] for idx in indices)

def extract_license_plates(data_dir, label_dir, output_dir, label_file, model_path='model.pth'):
    images, points = load(data_dir, label_dir)
    dataset = CustomDataset(images, points)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_data = []

    with torch.no_grad():
        for i, (images, points) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Change shape for cv2

            for j, (img, corners) in enumerate(zip(images, outputs)):
                original_img = (img * 255).astype(np.uint8)
                corners = corners.reshape(4, 2) * [img.shape[1], img.shape[0]]

                # Extract the license plate using perspective transform
                src_pts = corners.astype(np.float32)
                dst_pts = np.array([[0, 0], [256, 0], [256, 64], [0, 64]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                license_plate = cv2.warpPerspective(original_img, M, (256, 64))

                # Save the license plate image
                output_path = os.path.join(output_dir, f'plate_{i*32+j}.jpg')
                cv2.imwrite(output_path, license_plate)

                # Placeholder for actual Persian label extraction logic
                label = '۱۲۳۴۵۶۷۸'  # Replace this with the actual label extraction logic for Persian plates
                label_data.append(f'plate_{i*32+j}.jpg {label}')

    with open(label_file, 'w') as f:
        for item in label_data:
            f.write(f"{item}\n")

if __name__ == '__main__':
    data_dir = '/home/navid/Desktop/VisionProject/four-corners/images'
    label_dir = '/home/navid/Desktop/VisionProject/four-corners/labels'
    output_dir = '/home/navid/Desktop/VisionProject/four-corners/license_plates'
    label_file = '/home/navid/Desktop/VisionProject/four-corners/license_labels.txt'
    extract_license_plates(data_dir, label_dir, output_dir, label_file)
