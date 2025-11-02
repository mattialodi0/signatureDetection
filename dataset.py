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
from tqdm import tqdm
import webdataset as wds
from dataset_processing.data_augmentation import mirror_y, translate_y


def addDataset(dataset_path, name, source):
    img_files = []
    for img_file in os.listdir(os.path.join(datasets_paths, name, "images")):
        # sh.copy(os.path.join(datasets_paths, name, "images", img_file),
        #     os.path.join(dataset_path, "images", name[0]+img_file))
        # sh.copy(os.path.join(datasets_paths, name, "labels", img_file.replace(".png", ".txt").replace(".jpg", ".txt")),
        #     os.path.join(dataset_path, "labels", name[0]+img_file.replace(".png", ".txt").replace(".jpg", ".txt")))
        img_files.append(name[0]+img_file)

    # files = splitDataset(dataset_path, source["split"], img_files)
    files["train"] = augmentTrainDataset(
            os.path.join(dataset_path, "images", "train"), 
            os.path.join(dataset_path, "images", "train"),
            source.get("augmentations", {}), 
            files.get("train", []))
    return files

def augmentTrainDataset(image_dir, label_dir, augmentations, img_files):
    augmented_files = img_files.copy()
    for img_file in tqdm(img_files, desc="Applying augmentations"):
        for aug_name, aug_params in augmentations.items():
            prob = aug_params.get("probability", 1)
            if random.random() <= prob:
                if aug_name == "mirror_y":
                    mirror_y(image_dir, label_dir, img_file)
                elif aug_name == "translate_y":
                    translate_y(image_dir, label_dir, img_file, aug_params.get("probability", [0, 1]))
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
            }
        },
        "forms": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": []
        },
        "tobacco": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": []
        },
        "nist": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": []
        },
        "snps": {
            "split": [0.8, 0.1, 0.1],
            "augmentations": []
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
        break

    # files["train"].sort()
    # files["val"].sort()
    # files["test"].sort()

    # # from /images and /labels to a faster format
    # encodeDataset(dataset_path, images=files["train"], phase="train")
    # encodeDataset(dataset_path, images=files["val"], phase="val")
    # encodeDataset(dataset_path, images=files["test"], phase="test")

    # # files = os.listdir(os.path.join(dataset_path, "images", "test"))
