import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from regressor import CNNModel
from loader import load, CustomDataset
from torch.utils.data import DataLoader

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

def load_and_resize_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)
    return image_resized

def normalize_image(image):
    return image / 255.0

def corners(image: np.ndarray, model, device, target_size=(256, 256)) -> np.ndarray:
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, target_size)
    image_resized = normalize_image(image_resized)  # Normalize image
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = torch.tensor(image_resized, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    
    model.eval()
    with torch.no_grad():
        predicted_corners = model(image_resized).cpu().numpy()[0]

    width_ratio = original_size[1] / target_size[1]
    height_ratio = original_size[0] / target_size[0]
    predicted_corners[::2] *= width_ratio
    predicted_corners[1::2] *= height_ratio
    return predicted_corners

def calculate_accuracy(true_points, predicted_points, threshold=10):
    true_points = true_points.reshape(-1, 2) * 256
    predicted_points = predicted_points.reshape(-1, 2) * 256
    distances = np.sqrt(np.sum((true_points - predicted_points) ** 2, axis=1))
    accurate_predictions = np.mean(distances < threshold)
    return accurate_predictions

def visualize_results(images, true_points, predicted_points, save_path='visualizations'):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(images)):
        plt.figure()
        image_uint8 = (images[i].transpose(1, 2, 0) * 255).astype(np.uint8)  # Ensure correct shape
        if image_uint8.shape[2] == 1:  # Handle grayscale images
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        plt.imshow(cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB))
        tp = true_points[i].reshape(-1, 2) * 256
        pp = predicted_points[i].reshape(-1, 2) * 256
        plt.scatter(tp[:, 0], tp[:, 1], c='green', label='True Points')
        plt.scatter(pp[:, 0], pp[:, 1], c='red', label='Predicted Points')
        plt.title('True and Predicted Corners')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'result_{i}.png'))
        plt.close()

def evaluate_model(data_dir, label_dir, model_path='model.pth'):
    images, points = load(data_dir, label_dir)
    dataset = CustomDataset(images, points)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    all_predicted_points = []
    all_true_points = []

    with torch.no_grad():
        for images, points in test_loader:
            images, points = images.to(device), points.to(device)
            outputs = model(images)
            loss = criterion(outputs, points)
            test_loss += loss.item() * images.size(0)
            all_predicted_points.append(outputs.cpu().numpy())
            all_true_points.append(points.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    all_predicted_points = np.concatenate(all_predicted_points, axis=0)
    all_true_points = np.concatenate(all_true_points, axis=0)
    print(f"Test Loss: {test_loss:.4f}")

    accuracy = calculate_accuracy(all_true_points, all_predicted_points)
    print(f"Accuracy (distance < 10 pixels): {accuracy * 100:.2f}%")

    # Use the correct batch of images for visualization
    sample_images, _ = next(iter(test_loader))
    visualize_results(sample_images.cpu().numpy(), all_true_points, all_predicted_points)

if __name__ == '__main__':
    data_dir = '/home/navid/Desktop/VisionProject/four-corners/images'
    label_dir = '/home/navid/Desktop/VisionProject/four-corners/labels'
    evaluate_model(data_dir, label_dir)
