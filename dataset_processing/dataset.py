"""
    Customizable dataset creation module.
    possible sources: emails, forms, tobacco, nist, snps
    augmentation: mirror_x, mirror_y, translate_y

    labels are in the format: bbx bby bbw bbh
"""

import os
import shutil as sh
import json
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import webdataset as wds
from dataset_processing.data_augmentation import mirror_y, translate_y


def addDataset(dataset_path, name, source):
    img_files = []
    for img_file in os.listdir(os.path.join(datasets_paths, name, "images")):
        sh.copy(os.path.join(datasets_paths, name, "images", img_file),
                os.path.join(dataset_path, "images", name[0]+img_file))
        sh.copy(os.path.join(datasets_paths, name, "labels", img_file.replace(".png", ".txt").replace(".jpg", ".txt")),
                os.path.join(dataset_path, "labels", name[0]+img_file.replace(".png", ".txt").replace(".jpg", ".txt")))
        img_files.append(name[0]+img_file)

    files = splitDataset(dataset_path, source["split"], img_files)
    files["train"] = augmentTrainDataset(
        os.path.join(dataset_path, "images", "train"),
        os.path.join(dataset_path, "labels", "train"),
        source.get("augmentations", {}),
        files.get("train", []))
    return files

# WRONG naming logic
def augmentTrainDataset(image_dir, label_dir, augmentations, img_files):
    if not augmentations or len(augmentations.keys()) == 0:
        return img_files
    augmented_files = img_files.copy()
    for img_file in tqdm(img_files, desc="Applying augmentations"):
        # read image and labels
        img = cv2.imread(os.path.join(image_dir, img_file))
        labels = []
        with open(os.path.join(label_dir, img_file.replace(".png", ".txt").replace(".jpg", ".txt"))) as f:
            for line in f:
                labels.append(line.strip().split())
        # augmentations
        for aug_name, aug_params in augmentations.items():
            prob = aug_params.get("probability", 1)
            if random.random() <= prob:
                if not aug_params.get("replace", False):
                    img_file_aug = img_file.replace(".png", "_aug.png").replace(".jpg", "_aug.jpg")
                    cv2.imwrite(os.path.join(image_dir, img_file_aug), img)
                    with open(os.path.join(label_dir, img_file_aug.replace(".png", ".txt").replace(".jpg", ".txt")), 'w') as f:
                        for label in labels:
                            f.write(' '.join(map(str, label)) + '\n')
                    augmented_files.append(img_file_aug)
                if aug_name == "mirror_y":
                    img, labels = mirror_y(img, labels)
                elif aug_name == "translate_y":
                    range = aug_params.get("range", [0, 1])
                    rand_shift = random.randint(
                        int(range[0]*img.shape[0]), int(range[1]*img.shape[0]))
                    img, labels = translate_y(img, labels, rand_shift)                    
        # save augmented image and labels
        cv2.imwrite(os.path.join(image_dir, img_file), img)
        with open(os.path.join(label_dir, img_file.replace(".png", ".txt").replace(".jpg", ".txt")), 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    return augmented_files


def splitDataset(dataset_path, split, img_files):
    random.shuffle(img_files)
    train_images = img_files[:int(len(img_files) * split[0])]
    val_images = img_files[int(len(img_files) * split[0])
                               :int(len(img_files) * (split[0] + split[1]))]
    test_images = img_files[int(len(img_files) * (split[0] + split[1])):]
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    files = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for phase in ["train", "val", "test"]:
        if phase == "train":
            imgs = train_images
        elif phase == "val":
            imgs = val_images
        elif phase == "test":
            imgs = test_images

        for img in imgs:
            name = os.path.basename(img)
            lbl_name = name.replace('.jpg', '.txt').replace('.png', '.txt')
            os.rename(os.path.join(images_dir, img),
                      os.path.join(images_dir, phase, img))
            try:
                os.rename(os.path.join(labels_dir, lbl_name),
                          os.path.join(labels_dir, phase, lbl_name))
            except FileNotFoundError:
                print(f"Label file not found for image: {name}")
    return files


def encodeDataset(dataset_path, images, phase="train"):
    out_dir = os.path.join(dataset_path, f"{phase}-00000.tar")
    with wds.TarWriter(out_dir) as sink:
        for i, img in enumerate(tqdm(images, total=len(images))):
            ext = os.path.splitext(img)[1].lstrip(".").lower()
            img_path = os.path.join(dataset_path, "images", phase, img)
            label_path = os.path.join(dataset_path, "labels", phase, img.replace(
                '.jpg', '.txt').replace('.png', '.txt'))
            sample = {}
            base = os.path.splitext(os.path.basename(img_path))[0]

            if not os.path.exists(label_path):
                continue

            with open(label_path) as f:
                boxes = []
                for line in f:
                    x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])

            label_dict = {"boxes": boxes, "labels": [
                1] * len(boxes)}  # your class id(s)
            sample = {
                "__key__": base,
                ext.lstrip("."): open(img_path, "rb").read(),
                "json": json.dumps(label_dict),
            }
            sink.write(sample)


if __name__ == '__main__':
    datasets_paths = "datasets/base"
    dataset_path = "datasets/custom_augmented"
    sources = {
        "emails": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": {}
        },
        "forms": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": {}
        },
        "tobacco": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": {}
        },
        "nist": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": {
                "translate_y": {
                    "replace": True,
                    "probability": 1,
                    "range": [0.25, 1]
                },
                "mirror_y": {
                    "replace": False,
                    "probability": 1
                },
            },
        },
        "snps": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": {}
        }
    }

    assert all(sum(s["split"]) == 1.0 for s in sources.values())
    os.makedirs(os.path.join(dataset_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels", "test"), exist_ok=True)

    files = {}
    for source in sources.keys():
        fs = addDataset(dataset_path, source, sources[source])
        files["train"] = files.get("train", []) + fs["train"]
        files["val"] = files.get("val", []) + fs["val"]
        files["test"] = files.get("test", []) + fs["test"]

    files["train"].sort()
    files["val"].sort()
    files["test"].sort()

    # from /images and /labels to a faster format
    encodeDataset(dataset_path, images=files["train"], phase="train")
    encodeDataset(dataset_path, images=files["val"], phase="val")
    encodeDataset(dataset_path, images=files["test"], phase="test")
    

    # files = os.listdir(os.path.join(dataset_path, "images", "train"))
    # encodeDataset(dataset_path, images=files, phase="train")
    # files = os.listdir(os.path.join(dataset_path, "images", "val"))
    # encodeDataset(dataset_path, images=files, phase="val")
    # files = os.listdir(os.path.join(dataset_path, "images", "test"))
    # encodeDataset(dataset_path, images=files, phase="test")