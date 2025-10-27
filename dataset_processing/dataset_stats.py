import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_bbs(label_dir, image_dir, dim=(1000, 1414)):
    bbs = []
    files = []
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        x, y, w, h = map(int, parts)
                        if os.path.exists(os.path.join(image_dir, filename.replace('.txt', '.png'))):
                            img = cv2.imread(os.path.join(image_dir, filename.replace('.txt', '.png')))
                            ih, iw = img.shape[:2]
                            x,w,y,h = x / iw, w / iw, y / ih, h / ih
                            x,y,w,h = x * dim[0], y * dim[1], w * dim[0], h * dim[1]
                            bbs.append((int(x), int(y), int(w), int(h)))
                            files.append(filename)
                        elif os.path.exists(os.path.join(image_dir, filename.replace('.txt', '.jpg'))):
                            img = cv2.imread(os.path.join(image_dir, filename.replace('.txt', '.jpg')))
                            ih, iw = img.shape[:2]
                            x,w,y,h = x / iw, w / iw, y / ih, h / ih
                            x,y,w,h = x * dim[0], y * dim[1], w * dim[0], h * dim[1]
                            bbs.append((int(x), int(y), int(w), int(h)))
                            files.append(filename)
                        else:
                            print(f'Img: {filename} not found.')
        # if len(bbs) >= 100:
        #     break
    return bbs, files
            
for split in ['train', 'val', 'test']:
    print(f"--- {split} split ---")
    images_dir = f'datasets/custom_dataset_full/images/{split}'
    label_dir = f'datasets/custom_dataset_full/labels/{split}'
    bbs, files = get_bbs(label_dir, images_dir)

    def area(bb):
        _, _, w, h = bb
        return w * h
    areas = list(map(area, bbs))
    print(f'Mean area: {np.mean(areas)}')
    print(f'Min area: {np.min(areas)} for bb: {files[np.argmin(areas)]}')
    print(f'Max area: {np.max(areas)} for bb: {files[np.argmax(areas)]}')

    # Visualize bounding boxes
    img = np.full((int(1000*1.41), 1000, 3), 255, dtype=np.uint8)
    for bb in bbs:
        x, y, w, h = bb
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    plt.imshow(img)
    plt.title('Bounding Box Distribution')
    plt.axis('off')
    plt.show()