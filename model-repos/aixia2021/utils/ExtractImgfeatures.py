import argparse
import json
import os.path
from time import time

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.CNN import ResNet
from utils.wrap_var import to_var


def extract_features(img_dir, model, img_list, my_cpu=False):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if my_cpu:
        avg_img_features = np.zeros((5, 2048))
    else:
        avg_img_features = np.zeros((len(img_list), 2048))

    name2id = dict()
    print("extracting features ....")
    for i in range(len(img_list)):
        if i >= 5 and my_cpu:
            break
        ImgTensor = transform(
            Image.open(os.path.join(img_dir, img_list[i])).convert("RGB")
        )
        ImgTensor = to_var(ImgTensor.view(1, 3, 224, 224))
        conv_features, feat = model(ImgTensor)
        avg_img_features[i] = feat.cpu().data.numpy()
        name2id[img_list[i]] = i
    print("Done.")

    return avg_img_features, name2id

def get_image_data(n2n_file):
    data = []
    with open(n2n_file, "r") as file_v:
        n2n_data = json.load(file_v)
    for k, v in n2n_data.items():
        data.append(v["image_file"])
    return data

def create_image_features(image_dir, n2n_train_set, n2n_val_set, image_features_json_path, image_features_path):
    start = time()
    print("Creating new ResNet features")
    splits = ["train", "val"]

    my_cpu = False

    images = {"train": [], "val": []}

    images['train'] = get_image_data(n2n_train_set)
    images['val'] = get_image_data(n2n_val_set)

    model = ResNet()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    feat_h5_file = h5py.File(image_features_path, "w")
    json_data = dict()
    for split in splits:
        print(split)
        avg_img_features, name2id = extract_features(
            image_dir, model, img_list=images[split], my_cpu=my_cpu
        )
        feat_h5_file.create_dataset(
            name=split + "_img_features", dtype="float32", data=avg_img_features
        )
        json_data[split + "2id"] = name2id
    feat_h5_file.close()

    with open(image_features_json_path, "w") as f:
        json.dump(json_data, f)

    print("Image Features extracted.")
    print("Time taken: ", time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-image_dir",
        type=str,
        default="data/images",
        help="this directory should contain both train and val images",
    )
    parser.add_argument(
        "-n2n_train_set", type=str, default="data/n2n_train_successful_data.json"
    )
    parser.add_argument(
        "-n2n_val_set", type=str, default="data/n2n_val_successful_data.json"
    )
    parser.add_argument(
        "-image_features_json_path",
        type=str,
        default="data/ResNet_avg_image_features2id.json",
    )
    parser.add_argument(
        "-image_features_path", type=str, default="data/ResNet_avg_image_features.h5"
    )
    args = parser.parse_args()
    create_image_features(**vars(args))
