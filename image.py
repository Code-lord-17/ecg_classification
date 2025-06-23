import os
import piexif
from PIL import Image
import piexif.helper

# Base folder containing class folders
base_dir = "ecg_data"

# Get sorted list of class folders
folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

# Loop through folders and assign numeric labels
for idx, folder_name in enumerate(folders, start=1):  # Labels start from 1
    folder_path = os.path.join(base_dir, folder_name)
    print(f"Processing folder '{folder_name}' as label {idx}")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Open image
                img = Image.open(file_path)

                # Prepare new EXIF data
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
                label_str = str(idx)
                exif_dict["0th"][piexif.ImageIFD.Software] = label_str

                # Save image with EXIF, overwrite original
                exif_bytes = piexif.dump(exif_dict)
                img.save(file_path, exif=exif_bytes)
                print(f"✓ Updated {file_path} with label {label_str}")
            except Exception as e:
                print(f"✗ Failed to update {file_path}: {e}")
