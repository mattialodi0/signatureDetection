import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

imgs_dir = './datasets/base/tobacco/images'
labels_dir = './datasets/base/tobacco/labels'
annotations = './datasets/base/tobacco/labels.csv'

#  for classification
# p=0
# n=0
# for file in os.listdir('./datasets/Custom1/Form'):
#     if file.endswith('.jpg'):
#         img = cv2.imread(f"./datasets/Custom1/Form/{file}", cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img, cmap='gray')
#         plt.show()
#         label = input("press p to label as positive, n to label as negative and d to discard: ")
#         if label == 'd':
#             continue
#         elif label == 'p':
#             cv2.imwrite(f"./datasets/Custom1/Form/positive/{p}.jpg", img)
#             p+=1
#         elif label == 'n':
#             cv2.imwrite(f"./datasets/Custom1/Form/negative/{n}.jpg", img)
#             n+=1

# i=0
# for file in os.listdir('./datasets/custom1labelledcombined/Form/negative'):
#     os.rename(f"./datasets/custom1labelled/Tobacco/negative/{file}", f"./datasets/custom1labelled/Tobacco/positive/{i}.jpg")
#     i+=1

# --------------------

# for detection

os.makedirs(labels_dir, exist_ok=True)

df = pd.read_csv(annotations)

# COCO format: [x_min, y_min, width, height]
for img_file in os.listdir(imgs_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_name = os.path.splitext(img_file)[0]
    label_path = os.path.join(labels_dir, img_name + '.txt')
    # Filter annotations for this image
    boxes = df[df['image_name'] == img_file]
    with open(label_path, 'w') as f:
        for _, row in boxes.iterrows():
            x_min = row['bbox_x']
            y_min = row['bbox_y']
            width = row['bbox_width']
            height = row['bbox_height']
            f.write(f"{x_min} {y_min} {width} {height}\n")