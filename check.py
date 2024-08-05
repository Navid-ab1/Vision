import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from regressor import CNNModel
from loader import resize
from PIL import Image, ImageDraw

def load_model(model_path='model.pth'):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image, size=(224, 224)):
    resized_image, _ = resize(image, np.zeros(8), size)
    resized_image = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return resized_image.unsqueeze(0)

def de_normalize_points(points, original_size, resized_size=(224, 224)):
    original_width, original_height = original_size
    resized_width, resized_height = resized_size

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    de_normalized_points = points.copy()
    for i in range(4):
        de_normalized_points[2*i] *= scale_x
        de_normalized_points[2*i+1] *= scale_y

    return de_normalized_points

def predict_corners(image_path, model_path='model.pth'):
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    original_size = image.shape[1], image.shape[0]  # width, height
    preprocessed_image = preprocess_image(image).to(device)

    with torch.no_grad():
        output_points = model(preprocessed_image).squeeze(0).cpu().numpy()

    de_normalized_points = de_normalize_points(output_points, original_size)
    return image, de_normalized_points

def draw_corners(image, points):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    for i in range(4):
        x, y = points[2*i], points[2*i+1]
        draw.ellipse((x-5, y-5, x+5, y+5), outline='red', width=2)

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig('/home/navid/Desktop/output_image_display.jpg')  # Save the image to a file instead of showing it interactively
    print("Image with predicted corners displayed.")

if __name__ == '__main__':
    image_path = '/home/navid/Desktop/VisionProject/four-corners/images/1.jpg'  # Replace with the actual image path
    output_image_path = '/home/navid/Desktop/output_image.jpg'

    image, predicted_corners = predict_corners(image_path)
    if image is not None:
        image_with_corners = draw_corners(image, predicted_corners)
        cv2.imwrite(output_image_path, image_with_corners)
        print(f"Image with predicted corners saved to {output_image_path}")
        show_image(image_with_corners)
