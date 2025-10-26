import cv2
import numpy as np
import os
import random


def mirror_x(image_file):
    img_path = os.path.join(images_dir, image_file)
    img = cv2.imread(img_path)
    img_mirrored = cv2.flip(img, 0)
    image_path = os.path.join(output_images_dir, image_file.replace('.jpg', '_x_mirrored.jpg').replace('.png', '_x_mirrored.png'))
    cv2.imwrite(image_path, img_mirrored)

    label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(label_path):
        raise Exception('Label file does not exist')

    height = img.shape[0]
    with open(label_path, 'r') as f:
        lines = f.readlines()

    mirrored_lines = []
    for line in lines:
        parts = line.strip().split()

        x, y, w, h = parts
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        new_y = height - y - h
        mirrored_line = f"{x} {new_y} {w} {h}\n"
        mirrored_lines.append(mirrored_line)

    # Save mirrored labels
    mirrored_label_path = os.path.join(output_labels_dir, image_file.replace('.jpg', '_x_mirrored.txt').replace('.png', '_x_mirrored.txt'))
    with open(mirrored_label_path, 'w') as f:
        f.writelines(mirrored_lines)

def mirror_y(image_file):
    img_path = os.path.join(images_dir, image_file)
    img = cv2.imread(img_path)
    img_mirrored = cv2.flip(img, 1)
    image_path = os.path.join(output_images_dir, image_file.replace('.jpg', '_y_mirrored.jpg').replace('.png', '_y_mirrored.png'))
    cv2.imwrite(image_path, img_mirrored)

    label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(label_path):
        raise Exception('Label file does not exist')

    width = img.shape[1]
    with open(label_path, 'r') as f:
        lines = f.readlines()
    mirrored_lines = []
    for line in lines:
        parts = line.strip().split()
        x, y, w, h = parts
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        new_x = width - x - w
        mirrored_line = f"{new_x} {y} {w} {h}\n"
        mirrored_lines.append(mirrored_line)

    # Save mirrored labels
    mirrored_label_path = os.path.join(output_labels_dir, image_file.replace('.jpg', '_y_mirrored.txt').replace('.png', '_y_mirrored.txt'))
    with open(mirrored_label_path, 'w') as f:
        f.writelines(mirrored_lines)

def shift_y(image_file, shift_pixels):
    img_path = os.path.join(images_dir, image_file)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Wrap the bottom part to the top
    if shift_pixels > 0:
        # Take the bottom part and put it on top
        pad = img[-shift_pixels:, :]
        rest = img[:-shift_pixels, :]
        img_shifted = np.vstack((pad, rest))
    else:
        # Negative shift: take the top part and put it on bottom
        pad = img[:abs(shift_pixels), :]
        rest = img[abs(shift_pixels):, :]
        img_shifted = np.vstack((rest, pad))

    image_path = os.path.join(output_images_dir, image_file)
    cv2.imwrite(image_path, img_shifted)

    label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(label_path):
        raise Exception('Label file does not exist')

    with open(label_path, 'r') as f:
        lines = f.readlines()
    shifted_lines = []
    for line in lines:
        parts = line.strip().split()
        x, y, w, h = parts
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        # Wrap label y coordinate
        new_y = (y + shift_pixels) % height
        shifted_line = f"{x} {new_y} {w} {h}\n"
        shifted_lines.append(shifted_line)

    # Save shifted labels
    shifted_label_path = os.path.join(output_labels_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(shifted_label_path, 'w') as f:
        f.writelines(shifted_lines)

images_dir = './datasets/nist_copy/images'
labels_dir = './datasets/nist_copy/labels'
# output_images_dir = './datasets/nist_copy/images'
# output_labels_dir = './datasets/nist_copy/labels'
output_images_dir = './datasets/nist_copy/images_augmented'
output_labels_dir = './datasets/nist_copy/labels_augmented'
split = 1

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

img_files = os.listdir(images_dir)
random.shuffle(img_files)
selected_files = img_files[:int(len(img_files) * split)]

for img_file in selected_files:
    if not img_file.lower().endswith(('.jpg', '.png')):
        print("Skipping file:", img_file)
        continue
    # mirror_x(img_file)
    # mirror_y(img_file)
    rand_shift = random.randint(800, 3300)
    shift_y(img_file, rand_shift)