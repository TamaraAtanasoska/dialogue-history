import argparse
import json

import numpy as np

from lxmert.src.utils import load_obj_tsv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_mscoco_bottomup_train_tsv_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/train2014_obj36.tsv"
    )
    parser.add_argument(
        "--load_mscoco_bottomup_val_tsv_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/val2014_obj36.tsv"
    )
    parser.add_argument(
        "--save_mscoco_bottomup_info_json_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/mscoco_bottomup_info.json"
    )
    parser.add_argument(
        "--save_mscoco_bottomup_features_npy_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/save_mscoco_bottomup_features.npy"
    )
    parser.add_argument(
        "--save_mscoco_bottomup_boxes_npy_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/save_mscoco_bottomup_boxes.npy"
    )
    parser.add_argument("--breaking", type=bool, default=False)
    args = parser.parse_args()

    mscoco_bottomup_features = []

    print("\nLoading FasterRCNN features from: {}".format(args.load_mscoco_bottomup_train_tsv_filename))
    mscoco_bottomup_features.extend(
        load_obj_tsv(args.load_mscoco_bottomup_train_tsv_filename, topk=1000 if args.breaking else None))

    print("\nLoading FasterRCNN features from: {}".format(args.load_mscoco_bottomup_val_tsv_filename))
    mscoco_bottomup_features.extend(
        load_obj_tsv(args.load_mscoco_bottomup_val_tsv_filename, topk=1000 if args.breaking else None))

    image_id2image_pos = {}
    image_pos2image_id = {}
    mscoco_vectorized_features = []
    mscoco_vectorized_boxes = []
    mscoco_vectorized_img_h = []
    mscoco_vectorized_img_w = []
    for image_index, image_data in enumerate(mscoco_bottomup_features):
        mscoco_vectorized_features.append(image_data["features"])
        mscoco_vectorized_boxes.append(image_data["boxes"])
        mscoco_vectorized_img_h.append(image_data["img_h"])
        mscoco_vectorized_img_w.append(image_data["img_w"])
        image_id2image_pos[image_data["img_id"]] = image_index
        image_pos2image_id[image_index] = image_data["img_id"]

    mscoco_vectorized_features = np.array(mscoco_vectorized_features)
    mscoco_vectorized_boxes = np.array(mscoco_vectorized_boxes)

    print("Saving mscoco information to: {}".format(args.save_mscoco_bottomup_info_json_filename))
    with open(args.save_mscoco_bottomup_info_json_filename, mode="w") as out_file:
        json.dump(
            {
                "image_id2image_pos": image_id2image_pos,
                "image_pos2image_id": image_pos2image_id,
                "img_h": mscoco_vectorized_img_h,
                "img_w": mscoco_vectorized_img_w
            },
            out_file
        )

    print("Saving mscoco features to: {}".format(args.save_mscoco_bottomup_features_npy_filename))
    np.save(args.save_mscoco_bottomup_features_npy_filename, mscoco_vectorized_features)

    print("Saving mscoco boxes to: {}".format(args.save_mscoco_bottomup_boxes_npy_filename))
    np.save(args.save_mscoco_bottomup_boxes_npy_filename, mscoco_vectorized_boxes)
