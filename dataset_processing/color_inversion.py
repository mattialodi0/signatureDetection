import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

i=0
for file in os.listdir('./datasets/TobaccoInv'):
    if file.endswith('.png'):
        img = cv2.imread(f"./datasets/TobaccoInv/{file}", cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        inverted_img = cv2.bitwise_not(img)
        # plt.imshow(inverted_img, cmap='gray')
        # plt.show()
        cv2.imwrite(f"./datasets/Tobacco/{file}", inverted_img)
    if i%100==0:
        print(i)
    i+=1