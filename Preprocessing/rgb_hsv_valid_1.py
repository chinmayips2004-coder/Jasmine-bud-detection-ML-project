import cv2
import os

input_folder = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\valid\images"
output_folder = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\valid\images_hsv"

# Creating a folder for HSV images if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# Converting each image to HSV and saving it in the new folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(os.path.join(output_folder, filename), hsv)

print("HSV conversion complete for:", input_folder)

