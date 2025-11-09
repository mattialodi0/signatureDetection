import os, cv2

base_dir = "datasets/custom_augmented"
out_dir = "datasets/custom_augmented_yolo"
os.makedirs(out_dir, exist_ok=True)

for stage in ["train", "val", "test"]:
    label_dir = os.path.join(base_dir, "labels", stage)
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        try:
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r") as f:
                lines = f.readlines()

            image = None
            if os.path.exists(os.path.join(base_dir, "images", stage, label_file.replace(".txt", ".jpg"))):
                image = cv2.imread(os.path.join(base_dir, "images", stage, label_file.replace(".txt", ".jpg")))
            elif os.path.exists(os.path.join(base_dir, "images", stage, label_file.replace(".txt", ".png"))):
                image = cv2.imread(os.path.join(base_dir, "images", stage, label_file.replace(".txt", ".png")))
            else:
                print(f"Image for {label_file} not found, skipping.")
                continue
            img_h, img_w, _ = image.shape
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                bbox = list(map(int, parts))
                # Convert bbox from (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height)
                x_min, y_min, w, h = bbox
                x_center = x_min + w / 2
                y_center = y_min + h / 2
                new_bbox = [x_center/img_w, y_center/img_h, w/img_w, h/img_h] 
                new_line = f"{0} " + " ".join(f"{coord:.6f}" for coord in new_bbox)
                new_lines.append(new_line)
            
            label_path = os.path.join(out_dir, "labels", stage, label_file)
            with open(label_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
        except Exception as e:
            print(f"Error processing {label_file}: {e}")