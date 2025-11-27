# YOLOv8 Training and Visual Check
!pip install ultralytics

from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import os

# Dataset paths
train_images = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\train\images"
val_images   = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\valid\images"
test_images  = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\test\images"

# Creating dataset YAML
data_yaml_content = f"""
train: {train_images}
val: {val_images}
test: {test_images}

nc: 6
names: ['big bud', 'full bloom', 'medium bud', 'mediumsmall bud', 'pre bud stage', 'small bud']
"""

with open("jasmine_data.yaml", "w") as f:
    f.write(data_yaml_content)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='jasmine_data.yaml',
    epochs=100,
    imgsz=640,
    batch=8
)

# Evaluate model
model.val()

# Predict on test images
results = model.predict(
    source=test_images,
    conf=0.25,
    save=True
)

# Visual check of predictions
predicted_folder = 'runs/detect/predict'
predicted_images = [f for f in os.listdir(predicted_folder) if f.endswith(('.jpg', '.png'))][:3]

for img_file in predicted_images:
    img_path = os.path.join(predicted_folder, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Prediction: {img_file}')
    plt.show()

# Export weights
model.export(format='pt')
