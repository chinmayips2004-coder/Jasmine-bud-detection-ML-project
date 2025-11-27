import cv2
import os
import numpy as np

input_folder = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\test\images"
output_folder = r"D:\Chinmayi P S\project\Jasmine_bud_1.v3i.yolov8\test\images_clean"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Converting to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #Adjusted green range (to remove more leaf tones, but keep buds)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 220, 220])  # slightly less saturation/value to protect buds
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        #Morphological operations to smooth mask
        kernel = np.ones((3, 3), np.uint8)  # added the kernel
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)  # remove small dots
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)  # fill small holes

        # Inverting mask to keep buds
        mask_buds = cv2.bitwise_not(mask_green)

        # Applying mask
        result = cv2.bitwise_and(img, img, mask=mask_buds)

        # Making background white (for training clarity)
        white_bg = np.full_like(img, 255)
        final = np.where(result == 0, white_bg, result)

        cv2.imwrite(os.path.join(output_folder, filename), final)

print("Updated background removal complete for:", input_folder)
