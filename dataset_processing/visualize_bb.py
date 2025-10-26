import cv2
import numpy as np
import matplotlib.pyplot as plt


# Visualize a few validation images with predictions vs ground truth
def draw_boxes(img_tensor, boxes, color=(0,255,0), label=None, linewidth=2):
    img = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8).copy()
    for b in boxes:
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, linewidth)
    return img

name = 'r0500_01_y_mirrored'
img_path = f'./datasets/custom_dataset_augmented/images/train/{name}.png'
label_path = f'./datasets/custom_dataset_augmented/labels/train/{name}.txt'

plt.figure(figsize=(15,10))
cnt = 1
img = cv2.imread(img_path)
with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        bbx, bby, bbw, bbh = parts
        bbx, bby, bbw, bbh = int(bbx), int(bby), int(bbw), int(bbh)
        img_h, img_w = img.shape[:2]
        x2 = int(bbx+bbw)
        y2 = int(bby+bbh)
        cv2.rectangle(img, (bbx, bby), (x2, y2), (255, 0, 0), 5)
plt.axis('off'); 
plt.imshow(img)
plt.show()