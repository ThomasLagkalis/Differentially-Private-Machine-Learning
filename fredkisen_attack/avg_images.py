from model import load_orl_faces
import os
import cv2
import numpy as np

dataset_path = "data/"

def main():
    for person_id in range(40):
        folder = os.path.join(dataset_path, f"s{person_id+1}")
        
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        avg_img = np.zeros((112, 92), dtype=np.float32)
        img_count = 0
        
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
            if img is not None:
                # Resize to standard ORL size if needed
                if img.shape != (112, 92):
                    img = cv2.resize(img, (92, 112))
                img = img.astype(np.float32)
                avg_img += img
                img_count += 1
        
        if img_count > 0:
            avg_img = avg_img/img_count
            #print(np.max(avg_img))
            # Convert back to uint8 for saving
            avg_img = avg_img.astype(np.uint8)
            
            output_path = os.path.join("data/avg_images/", f"average_face{person_id}.png")
            cv2.imwrite(output_path, avg_img)
            print(f"Saved average face for person {person_id+1} to {output_path}")
        else:
            print(f"No images found in {folder}")

if __name__ == "__main__":
    main()
