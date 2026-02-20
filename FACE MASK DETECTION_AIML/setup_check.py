import os

def check_structure():
    required_dirs = ['dataset', 'models', 'src', 'templates']
    missing_dirs = []
    
    base_path = os.getcwd()
    print(f"Checking project structure in: {base_path}")
    
    for d in required_dirs:
        if not os.path.exists(d):
            missing_dirs.append(d)
            try:
                os.makedirs(d)
                print(f"Created missing directory: {d}")
            except OSError as e:
                print(f"Error creating directory {d}: {e}")
        else:
            print(f"Directory exists: {d}")
            
    # Check dataset specific structure
    if os.path.exists('dataset'):
        subdirs = ['with_mask', 'without_mask']
        for sd in subdirs:
            path = os.path.join('dataset', sd)
            if not os.path.exists(path):
                print(f"WARNING: Dataset subdirectory '{sd}' missing in 'dataset/'")
                try:
                    os.makedirs(path)
                    print(f"Created placeholder directory: {path}")
                except OSError as e:
                    print(f"Error creating directory {path}: {e}")
            else:
                # Check for images
                files = os.listdir(path)
                image_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"Dataset '{sd}': {image_count} images found.")
                
                if image_count == 0:
                    print(f"  -> WARNING: No images found in {path}. Please add images.")

if __name__ == "__main__":
    check_structure()
