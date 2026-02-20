from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import winsound
import time

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join('models', 'mask_detector.h5')
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

print("Loading model...")
try:
    maskNet = load_model(MODEL_PATH)
    print("Model loaded.")
except:
    print("Model not found. Please train model first.")
    maskNet = None

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if maskNet is not None:
                # Mirror
                frame = cv2.flip(frame, 1)
                
                # Preprocess
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                
                for (x, y, w, h) in faces:
                    face_roi = rgb_frame[y:y+h, x:x+w]
                    try:
                        face_roi = cv2.resize(face_roi, (224, 224))
                        face_roi = tf.keras.preprocessing.image.img_to_array(face_roi)
                        face_roi = face_roi / 255.0
                        face_roi = np.expand_dims(face_roi, axis=0)
                        
                        (withoutMask, withMask) = maskNet.predict(face_roi)[0]
                        
                        label = "Mask" if withMask > withoutMask else "No Mask"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                        
                        if label == "No Mask":
                            # Beep in background thread or very short beep to not block stream
                            # winsound.Beep(1000, 100) # Can cause lag in stream
                            pass 

                        label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)

                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    except:
                        pass
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
