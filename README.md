# Jasmine-bud-detection-ML-project
Jasmine bud stage detection - preprocessing, CV, and ML models.

## Project Overview

This repository contains the work completed on jasmine bud image analysis and classification. It includes preprocessing scripts, RGB model training code, dataset samples, and progress documentation.

## Work Completed

- Captured and organised jasmine bud images under natural, real-world conditions.

- Performed RGB image preprocessing, including resizing, contrast adjustments, and noise-handling steps.

- Converted images to HSV format to improve separation of flower vs. background owing to clear color distinctions in HSV space.

- Prepared datasets for YOLO-style annotations and trained an RGB-only classification model.

- Achieved a maximum accuracy of 53% using RGB images (HSV-based training pending).

- Documented training results and current project status in the progress report.

- Uploaded all Python scripts, notebooks, and dataset samples for reproducibility.

## Reason for HSV Conversion

- HSV is preferred over formats like YCbCr or Lab because hue provides stronger separation between the green background and white/cream bud region.

- Reduces sensitivity to illumination changes, which is important for outdoor plant images.

- Simplifies segmentation due to distinct color space distribution.

## Repository Contents

- Preprocessing scripts - RGB to HSV conversion, morphological operations

- Training scripts/notebooks - RGB training notebook, logs

- Dataset samples - Raw + processed images

- Progress report - PDF documenting work completed

- Requirements.txt - List of required libraries
