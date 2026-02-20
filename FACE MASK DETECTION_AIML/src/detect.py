import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import winsound  # Specific to Windows for sound alert

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = tf.keras.applications.mobilenet_v2.preprocess_input(face) # Or just /255.0 depending on training
                # Since we trained with /255.0 in preprocess.py:
                # face = face / 255.0 
                # HOWEVER, MobileNetV2 usually expects specific preprocessing if using transfer learning. 
                # Our simple CNN used 1.0/255.0 rescale.
                
                # Check consistency with train.py
                # train.py uses: rescale=1.0/255.0
                face = face / 255.0
                
                face = np.expand_dims(face, axis=0)

                faces.append(face)
                locs.append((startX, startY, endX, endY))
            except Exception as e:
                pass

    if len(faces) > 0:
        preds = maskNet.predict(faces)

    return (locs, preds)

def start_detection():
    # Load Face Detector (Caffe model for better accuracy than Haar)
    # If not available, we can fallback to Haar. 
    # For now, let's use Haar as requested in the Prompt ("Face Detection using Haar Cascades")
    # BUT user prompt also mentioned "Requirements: opencv-python". 
    # Let's stick to Haar as requested in steps.
    
    # HAAR CASCADE IMPLEMENTATION
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load Mask Detector Model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'mask_detector.h5')
    if not os.path.exists(model_path):
        print("Model not found! Please train the model first.")
        return
        
    print("Loading model...")
    maskNet = load_model(model_path)

    print("Starting video stream...")
    vs = cv2.VideoCapture(0)

    while True:
        ret, frame = vs.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For CNN
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # ROI for prediction
            face_roi = rgb_frame[y:y+h, x:x+w]
            try:
                face_roi = cv2.resize(face_roi, (224, 224))
                face_roi = tf.keras.preprocessing.image.img_to_array(face_roi)
                face_roi = face_roi / 255.0 # Normalize
                face_roi = np.expand_dims(face_roi, axis=0)
                
                (withoutMask, withMask) = maskNet.predict(face_roi)[0]
                
                label = "Mask" if withMask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                # Alert Logic
                if label == "No Mask":
                    winsound.Beep(1000, 200) # Frequency 1000Hz, Duration 200ms
                
                label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)
                
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
            except Exception as e:
                pass

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()
