# Kidney Tumor Detection and Real-Time Testing

## Overview
This project involves the detection of kidney tumors using deep learning models. It consists of two main workflows:

1. **Training a model on Kaggle:** A notebook is provided that trains the detection model on the Kaggle platform using kidney tumor datasets.
2. **Real-time testing on local systems:** A second notebook demonstrates how to use the trained model weights for real-time detection and includes additional features like percentage-based metrics to evaluate the results.

The system employs YOLO-based object detection methods to identify and analyze kidney tumors in medical images, ensuring both accuracy and usability for real-time scenarios.

---

## File Structure
```
project_directory/
├── kidney_tumor_detection.ipynb      # Training notebook on Kaggle
├── updated_with_percentage.ipynb     # Real-time testing notebook on local system
├── weights/                          # Directory containing trained model weights
│   ├── best.pt                       # YOLO weights for kidney tumor detection
└── data/                             # Dataset and testing images
```

---

## Dependencies
### Required Libraries
Ensure the following libraries are installed:
- Python >= 3.8
- Jupyter Notebook
- YOLO (via `ultralytics`)
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Pandas
- Torch

Install the dependencies using:
```bash
pip install ultralytics opencv-python numpy matplotlib pandas torch
```

---

## 1. Training Notebook (Kaggle)
### Description
The `kidney_tumor_detection.ipynb` notebook is designed to train a YOLO model for kidney tumor detection using datasets available on Kaggle.

### Steps:
1. **Dataset Preparation:**
   - The dataset is downloaded and split into training and validation sets.
   - Images are annotated for kidney tumor regions.

2. **Model Training:**
   - YOLOv8 is used for object detection training.
   - Key metrics such as precision, recall, and mAP are calculated.

3. **Saving Weights:**
   - The best-performing model is saved as `best.pt` for inference.

### Outputs:
- Trained weights (`best.pt`).
- Performance metrics and loss curves.

---

## 2. Real-Time Testing Notebook (Local)
### Description
The `updated_with_percentage.ipynb` notebook demonstrates real-time tumor detection on local systems using the trained model weights.

### Features:
1. **Model Loading:**
   - Loads the `best.pt` weights from the training phase.

2. **Real-Time Detection:**
   - Processes images or video feeds to detect kidney tumors.
   - Outputs annotated images with bounding boxes around detected regions.

3. **Percentage-Based Metrics:**
   - Calculates detection percentages based on tumor size or region.
   - Provides detailed analysis for each frame or image.

### Outputs:
- Annotated images or videos.
- Detection statistics, including tumor size percentages.

---

## Usage
1. **Training (Optional):**
   - Open `kidney_tumor_detection.ipynb` in Kaggle.
   - Run the notebook to train the YOLO model on the dataset.
   - Download the `best.pt` file after training.

2. **Testing:**
   - Place `best.pt` in the `weights/` directory.
   - Open `updated_with_percentage.ipynb` on a local machine.
   - Adjust file paths and folder locations as needed.
   - Run the notebook to test real-time kidney tumor detection.

---

## Key Functions
### Training Notebook:
- **`train_yolo`**: Handles model training.
- **`evaluate_model`**: Computes precision, recall, and mAP.

### Testing Notebook:
- **`load_model`**: Loads the trained YOLO weights.
- **`detect_and_annotate`**: Runs detection and annotates images.
- **`calculate_percentage`**: Computes tumor detection percentages.

---

## Notes
- The dataset must be preprocessed and annotated correctly for optimal performance.
- For real-time detection, ensure sufficient system resources to handle video processing.

---

## Future Improvements
- Extend detection capabilities to 3D medical imaging.
- Automate dataset annotation using semi-supervised learning.
- Deploy the model as a web or mobile application for broader accessibility.

---


