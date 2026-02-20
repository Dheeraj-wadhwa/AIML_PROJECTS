# Face Mask Detection with Live Alert System

## 1. Introduction
The COVID-19 pandemic has made wearing face masks a critical safety measure in public spaces. Monitoring compliance manually is labor-intensive and risky for human inspectors. This project aims to automate this process using computer vision and deep learning. We have developed a real-time Face Mask Detection System that identifies whether a person is wearing a mask or not via a webcam feed and triggers an immediate audio-visual alert for non-compliance. This solution can be deployed in hospitals, airports, offices, and schools to ensure public safety.

## 2. Abstract
This project implements a lightweight yet robust system for detecting face masks in real-time video streams. The system utilizes a two-stage architecture: first, **Haar Cascade Classifiers** are used for efficient face detection to localize faces within a video frame. Second, a **Convolutional Neural Network (CNN)**—built with TensorFlow and Keras—analyzes the cropped face regions to classify them into two categories: "Mask" or "No Mask". The system is integrated with **OpenCV** for video processing and **Flask** for web-based deployment. Experimental results on a balanced dataset show high accuracy, processing frames with low latency suitable for real-world deployment.

## 3. Tools Used
The following technologies and libraries were utilized to build this system:
- **Python 3.x**: The core programming language.
- **TensorFlow / Keras**: For building, training, and saving the Convolutional Neural Network (CNN) model.
- **OpenCV (cv2)**: For real-time video capturing, image processing (grayscale conversion, resizing), and drawing bounding boxes.
- **Haar Cascades**: A pre-trained machine learning object detection method used for identifying frontal faces.
- **Flask**: A micro web framework used to stream the video feed to a web browser.
- **Numpy**: For numerical array operations.
- **Winsound**: For generating system audio alerts on Windows.
- **Start-of-the-art IDE**: VS Code for development and debugging.

## 4. Steps Involved in Building the Project

### Step 1: Data Collection & Preprocessing
- **Data Collection**: We created a custom data capture tool (`src/capture_data.py`) to gather images of faces "with_mask" and "without_mask" using a webcam.
- **Preprocessing**: Images were resized to **224x224** pixels to match the input layer of our CNN. Pixel values were normalized (scaled between 0 and 1) to aid model convergence.
- **Augmentation**: To prevent overfitting, we applied data augmentation techniques such as rotation, zooming, and horizontal flipping using `ImageDataGenerator`.

### Step 2: Model Architecture Design
We designed a sequential CNN model (`src/train.py`) optimized for image classification:
- **Convolutional Layers**: Three blocks of Conv2D layers (32, 64, 128 filters) with ReLU activation to feature extraction.
- **Pooling Layers**: MaxPooling2D layers to reduce spatial dimensions and computation.
- **Flattening**: Converting the 2D feature maps into a 1D vector.
- **Dropout**: A dropout rate of 0.5 was applied to randomly drop neurons during training, reducing overfitting.
- **Output Layer**: A Dense layer with Softmax activation to output probabilities for the two classes.

### Step 3: Model Training
- The model was compiled using the **Adam** optimizer and **Categorical Crossentropy** loss function.
- We trained the model on our dataset (split into training and validation sets) and monitored accuracy and loss.
- The trained model was saved as `models/mask_detector.h5` for inference.

### Step 4: Real-Time Detection Logic
- An inference script (`src/detect.py`) was created to capture video frames.
- **Pipeline**:
    1.  Detect faces using Haar Cascade.
    2.  Extract the Face ROI (Region of Interest).
    3.  Preprocess the ROI (Resize -> Normalize).
    4.  Pass ROI to the loaded CNN model for prediction.
    5.  Based on the prediction confidence, assign a label ("Mask" / "No Mask").

### Step 5: Implementation of Alert System & Deployment
- **Alerts**: If "No Mask" is detected, the system draws a **Red** bounding box and plays a sound alert (`winsound.Beep`). If a mask is detected, a **Green** box is shown.
- **Web App**: We wrapped the detection logic in a **Flask** application (`app.py`), enabling the video feed to be viewed accessible via a web browser (`http://127.0.0.1:5000`).

## 5. Conclusion
We successfully developed and deployed a generic Face Mask Detection System. The project demonstrates the power of combining classical computer vision (Haar Cascades) with modern deep learning (CNNs) to solve relevant real-world problems. The system is accurate, works in real-time, and includes a user-friendly alert mechanism. Future scopes include deploying the model to edge devices like Raspberry Pi and improving face detection robustness using MTCNN or SSD models.
