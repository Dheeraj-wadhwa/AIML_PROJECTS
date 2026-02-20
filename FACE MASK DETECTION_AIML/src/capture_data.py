import cv2
import os
import time

def capture_images(label, num_samples=100):
    """
    Captures images from the webcam and saves them to the specified label directory.
    label: 'with_mask' or 'without_mask'
    num_samples: Number of images to capture
    """
    save_path = os.path.join("dataset", label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Haar Cascade for face detection (used to crop face)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    print(f"Starting capture for '{label}'. Look at the camera!")
    print("Press 'q' to quit early.")
    time.sleep(2)  # Give user time to prepare

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Draw rectangle for visual feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save the face ROI
            face_roi = frame[y:y+h, x:x+w]
            try:
                # Resize to model input size (optional here, but good practice)
                face_roi = cv2.resize(face_roi, (224, 224))
                file_name = os.path.join(save_path, f"{label}_{count}.jpg")
                cv2.imwrite(file_name, face_roi)
                count += 1
                print(f"Captured {count}/{num_samples}", end='\r')
            except Exception as e:
                print(f"Error saving image: {e}")

            # Only capture one face per frame to avoid duplicates/confusion
            break

        cv2.imshow(f"Capturing {label}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture complete for {label}. Saved {count} images.")

if __name__ == "__main__":
    print("Select mode:")
    print("1. Capture 'with_mask' images (Wear a mask!)")
    print("2. Capture 'without_mask' images (Don't wear a mask!)")
    
    choice = input("Enter choice (1/2): ")
    
    if choice == '1':
        capture_images('with_mask')
    elif choice == '2':
        capture_images('without_mask')
    else:
        print("Invalid choice.")
