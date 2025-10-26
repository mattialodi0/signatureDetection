import os
from PIL import Image

# === CONFIGURATION ===
IMG_DIR = "datasets/custom_dataset/images"          # path to your images
YOLO_LABEL_DIR = "datasets/custom_dataset/labels"  # path to your original YOLO .txt files
VOC_LABEL_DIR = "datasets/custom_dataset/annotations"    # output path for converted labels
IMG_EXT = ".jpg"                    # or ".jpg", ".jpeg"

os.makedirs(VOC_LABEL_DIR, exist_ok=True)

def convert_yolo_to_voc_line(yolo_line, img_width, img_height):
    """
    Convert a single YOLO label line to VOC format.
    YOLO: class x_center y_center width height (normalized)
    VOC:  x_min y_min x_max y_max (absolute)
    """
    parts = yolo_line.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Invalid YOLO label line: {yolo_line}")
    x_c, y_c, w, h = map(float, parts)
    x_min = (x_c - w / 2) * img_width
    y_min = (y_c - h / 2) * img_height
    x_max = (x_c + w / 2) * img_width
    y_max = (y_c + h / 2) * img_height
    return f"{x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n"

def main():
    yolo_files = [f for f in os.listdir(YOLO_LABEL_DIR) if f.endswith(".txt")]
    print(f"Found {len(yolo_files)} YOLO label files")

    for yolo_file in yolo_files:
        img_name = os.path.splitext(yolo_file)[0] + IMG_EXT
        img_path = os.path.join(IMG_DIR, img_name)
        yolo_path = os.path.join(YOLO_LABEL_DIR, yolo_file)
        voc_path = os.path.join(VOC_LABEL_DIR, yolo_file)

        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {yolo_file}: image not found ({img_path})")
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            W, H = img.size

        # Convert all lines
        with open(yolo_path, "r") as f_in, open(voc_path, "w") as f_out:
            for line in f_in:
                try:
                    voc_line = convert_yolo_to_voc_line(line, W, H)
                    f_out.write(voc_line)
                except Exception as e:
                    print(f"Error converting line in {yolo_file}: {e}")

    print(f"✅ Conversion complete. VOC labels saved in: {VOC_LABEL_DIR}")

if __name__ == "__main__":
    main()
