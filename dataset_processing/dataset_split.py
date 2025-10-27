import os 
import random


images_dir = './datasets/custom_dataset_augmented/images/'
labels_dir = './datasets/custom_dataset_augmented/labels/'
os.makedirs(f"{images_dir}/val", exist_ok=True)
os.makedirs(f"{labels_dir}/val", exist_ok=True)
os.makedirs(f"{images_dir}/test", exist_ok=True)
os.makedirs(f"{labels_dir}/test", exist_ok=True)
images_dir = './datasets/custom_dataset_augmented/images/train'
labels_dir = './datasets/custom_dataset_augmented/labels/train'

train_split = 0.8
val_split = 0.1
test_split = 0.1
img_files = os.listdir(images_dir)
random.shuffle(img_files)
train_images = img_files[:int(len(img_files) * train_split)]
val_images = img_files[int(len(img_files) * train_split):int(len(img_files) * (train_split + val_split))]
test_images = img_files[int(len(img_files) * (train_split + val_split)):]

for img in val_images:
    name = os.path.basename(img)
    lbl_name = name.replace('.jpg', '.txt').replace('.png', '.txt')
    os.rename(f"{images_dir}/{name}", f"{images_dir.replace('train', 'val')}/{name}")
    try:
        os.rename(f"{labels_dir}/{lbl_name}", f"{labels_dir.replace('train', 'val')}/{lbl_name}")
    except FileNotFoundError:
        print(f"Label file not found for image: {name}")

for img in test_images:
    name = os.path.basename(img)
    lbl_name = name.replace('.jpg', '.txt').replace('.png', '.txt')
    os.rename(f"{images_dir}/{name}", f"{images_dir.replace('train', 'test')}/{name}")
    try:
        os.rename(f"{labels_dir}/{lbl_name}", f"{labels_dir.replace('train', 'test')}/{lbl_name}")
    except FileNotFoundError:
        print(f"Label file not found for image: {name}")
