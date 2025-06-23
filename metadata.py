import piexif
from PIL import Image
import piexif.helper

def extract_predict(image_path):
    try:
        img = Image.open(image_path)
        
        # Check for image format
        if img.format not in ["JPEG", "JPG"]:
            return {
                "value": "Unknown",
                "error": f"Unsupported format: {img.format}"
            }

        exif_data = img.info.get("exif")
        if not exif_data:
            return {
                "value": "Unknown"
            }

        exif_dict = piexif.load(exif_data)

        software = exif_dict["0th"].get(piexif.ImageIFD.Software, b"").decode(errors='ignore')
        if software == "1":
            software = "mental stress"
        elif software == "2":   
            software = "no stress"
        elif software == "3":
            software = "physical stress"
        return {
            "value": software if software else "Unknown",
        }
    except Exception as e:
        return {
            "value": "Unknown"
        }
