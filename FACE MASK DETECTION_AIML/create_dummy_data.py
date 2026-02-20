import cv2
import numpy as np
import os

def create_dummy():
    dirs = ['dataset/with_mask', 'dataset/without_mask']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
        
        # Check if empty
        if len(os.listdir(d)) == 0:
            print(f"Creating dummy images in {d}...")
            # Create 5 dummy images
            for i in range(5):
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                # Make 'without_mask' distinct (white vs black)
                if 'without' in d:
                    img.fill(255)
                
                cv2.imwrite(os.path.join(d, f"dummy_{i}.jpg"), img)

if __name__ == "__main__":
    create_dummy()
