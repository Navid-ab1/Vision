import os
import torch
import numpy as np
import cv2
from regressor import CNNModel
from classifier import LicensePlateClassifier

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

def corners_from_xml(xml_path, img_width, img_height):
    objects = parse_xml(xml_path, img_width, img_height)
    # Assuming the corners are the last object in the XML
    if objects:
        class_id, xmin, ymin, xmax, ymax = objects[-1]
        corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        return corners
    else:
        return None

def corners(image, xml_path, device):
    img_height, img_width = image.shape[:2]
    corners_points = corners_from_xml(xml_path, img_width, img_height)
    if corners_points is None:
        return None
    return corners_points

def load_and_resize_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    return image

def load_persian_characters():
    persian_chars = '۱۲۳۴۵۶۷۸۹۰' + 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
    return persian_chars

def map_to_persian_chars(indices):
    persian_chars = load_persian_characters()
    return ''.join(persian_chars[idx] for idx in indices)

def read_plate(image, corners_points, classifier, device):
    if corners_points is None:
        return "Error: No corners detected"
    
    src_pts = corners_points.astype(np.float32)
    dst_pts = np.array([[0, 0], [256, 0], [256, 64], [0, 64]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    license_plate = cv2.warpPerspective(image, M, (256, 64))

    license_plate_resized = cv2.resize(license_plate, (128, 64)) / 255.0
    license_plate_resized = torch.tensor(license_plate_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    classifier.eval()
    with torch.no_grad():
        outputs = classifier(license_plate_resized)
    predicted_chars = outputs.argmax(dim=-1).cpu().numpy().flatten()

    return map_to_persian_chars(predicted_chars)

def read_single(image_path, xml_path, regressor_path='model.pth', classifier_path='classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    regressor = CNNModel()
    regressor.load_state_dict(torch.load(regressor_path))
    regressor.to(device)
    
    classifier = LicensePlateClassifier()
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.to(device)
    
    image = load_and_resize_image(image_path)
    corners_points = corners(image, xml_path, device)
    plate_content = read_plate(image, corners_points, classifier, device)
    
    return plate_content

def process_directory(base_dir, regressor_path='model.pth', classifier_path='classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    regressor = CNNModel()
    regressor.load_state_dict(torch.load(regressor_path))
    regressor.to(device)
    
    classifier = LicensePlateClassifier()
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.to(device)

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            imgs_dir = os.path.join(subdir_path, 'imgs')
            anns_dir = os.path.join(subdir_path, 'anns')
            if os.path.exists(imgs_dir) and os.path.exists(anns_dir):
                for img_file in os.listdir(imgs_dir):
                    if img_file.endswith(".jpg"):
                        image_path = os.path.join(imgs_dir, img_file)
                        xml_path = os.path.join(anns_dir, img_file.replace(".jpg", ".xml"))
                        if os.path.exists(xml_path):
                            plate_content = read_single(image_path, xml_path, regressor, classifier, device)
                            print(f"Image: {image_path}, License Plate Content: {plate_content}")

if __name__ == '__main__':
    base_dir = '/path/to/base/directory'
    process_directory(base_dir)
